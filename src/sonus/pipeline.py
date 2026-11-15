"""
Cascaded asynchronous event-driven voice pipeline orchestrator.

Coordinates bidirectional acoustic streaming across discrete sub-components:
Silero VAD v5 -> faster-whisper CTranslate2 STT -> Ollama LLM -> Kokoro-82M ONNX TTS.
Enforces thread-safe barge-in preemption and serializes multi-turn dialogue context.
"""

import asyncio
import json
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from sonus.config import SonusConfig
from sonus.audio.stream import AudioStreamManager
from sonus.audio.ring_buffer import AudioRingBuffer
from sonus.vad.silero import SileroVAD
from sonus.stt.transcriber import WhisperTranscriber
from sonus.llm.client import StreamingLLMClient
from sonus.tts.synthesizer import KokoroSynthesizer


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy scalars and non-finite floats to JSON-serializable types (RFC 8259)."""
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if hasattr(obj, "_mock_name") or type(obj).__name__ in ("MagicMock", "Mock"):
        return str(obj)
    if isinstance(obj, (bool, str, type(None))):
        return obj
    return str(obj)


class VoicePipeline:
    """
    State machine orchestrator managing continuous voice conversational interaction.
    
    Architectural State Model:
        - READY: Audio streams active, VAD listening.
        - LISTENING: User speech detected, acoustic chunks accumulating.
        - TRANSCRIBING: Trailing silence detected; chunk dispatched to faster-whisper on thread pool.
        - THINKING: Transcription delivered; Ollama token streaming initiated.
        - SPEAKING: First sentence boundary extracted; Kokoro TTS synthesizing audio frames.
        - BARGE_IN: User vocalization detected while speaker active; immediate preemption dispatched.
    """

    def __init__(
        self,
        config: SonusConfig,
        on_status: Optional[Callable[[str, str], None]] = None,
        on_transcription: Optional[Callable[[str], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
        on_level: Optional[Callable[[float], None]] = None,
    ):
        self.config = config
        self.on_status = on_status or (lambda status, details: None)
        self.on_transcription = on_transcription or (lambda text: None)
        self.on_response = on_response or (lambda text: None)
        self.on_level = on_level

        # Components
        self.audio = AudioStreamManager(config.audio)
        self.vad = SileroVAD(config.vad, sample_rate=config.audio.input_sample_rate)
        self.stt = WhisperTranscriber(config.stt)
        self.llm = StreamingLLMClient(config.llm)
        self.tts = KokoroSynthesizer(config.tts)

        # State tracking
        # Ring buffer holds ~300ms of pre-speech audio so the beginning of words aren't clipped
        pre_speech_chunks = int(300 / config.audio.frame_size_ms)
        self._pre_speech_buffer = AudioRingBuffer(max_chunks=max(1, pre_speech_chunks))
        self._speech_chunks: List[np.ndarray] = []

        self._is_speaking_user = False
        self._silence_start_time: Optional[float] = None
        self._running = False
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._processing_task: Optional[asyncio.Task] = None
        self._utterance_lock: asyncio.Lock = asyncio.Lock()
        self._conversation_history: List[dict] = []
        self._debug_record_count: int = 0

    def _cancel_processing_task(self) -> None:
        """Safely cancel the active processing task on the event loop."""
        if self._event_loop and not self._event_loop.is_closed() and self._event_loop.is_running():
            def _cancel():
                if self._processing_task is not None and not self._processing_task.done():
                    self._processing_task.cancel()
            self._event_loop.call_soon_threadsafe(_cancel)
        elif self._processing_task is not None and not self._processing_task.done():
            self._processing_task.cancel()

    def _schedule_utterance_processing(self, audio_data: np.ndarray) -> None:
        """Schedule utterance handling on the event loop, tracking the active Task."""
        if not self._running or self._event_loop is None or not self._event_loop.is_running():
            return

        def _create_task():
            if self._processing_task is not None and not self._processing_task.done():
                self._processing_task.cancel()
            task = self._event_loop.create_task(
                self._handle_user_utterance(audio_data)
            )
            def _done_cb(t: asyncio.Task) -> None:
                if not t.cancelled():
                    try:
                        t.exception()
                    except (asyncio.CancelledError, Exception):
                        pass
            task.add_done_callback(_done_cb)
            self._processing_task = task

        self._event_loop.call_soon_threadsafe(_create_task)

    def _audio_callback(self, chunk: np.ndarray) -> None:
        """Invoked on audio thread for every incoming 32ms microphone frame."""
        if not self._running or self._event_loop is None:
            return

        is_speech, score = self.vad.is_speech(chunk)

        # 1. Echo suppression & Barge-in detection during playback
        if self.audio.is_playing:
            # Echo suppression: do not pollute pre-speech buffer with speaker output
            self._pre_speech_buffer.clear()

            # Barge-in requires higher speech confidence to prevent self-interruption from speaker echo
            barge_in_threshold = max(self.config.vad.threshold + 0.25, 0.65)
            if is_speech and score >= barge_in_threshold:
                self.on_status("BARGE_IN", "User interruption detected; interrupting playback.")
                self.audio.stop_playback()
                self._cancel_processing_task()
                self._is_speaking_user = True
                self._speech_chunks = [chunk]
                self._silence_start_time = None
                self.vad.reset_states()
                # Maintain the trailing 64 samples of this chunk as context for the next frame
                self.vad._context = chunk[-self.vad._context_size:].copy()
            return

        # 2. User Speech Lifecycle
        now = time.monotonic()
        if is_speech:
            self._silence_start_time = None
            if not self._is_speaking_user:
                # Speech just started
                self._is_speaking_user = True
                self.on_status("LISTENING", "User started speaking...")
                # Prepend the pre-speech ring buffer so we don't cut off consonants
                pre_speech = self._pre_speech_buffer.get_all()
                self._speech_chunks = [pre_speech] if pre_speech.size > 0 else []

            self._speech_chunks.append(chunk)
        else:
            if self._is_speaking_user:
                # Still within a speech turn, but this frame is silent
                self._speech_chunks.append(chunk)
                if self._silence_start_time is None:
                    self._silence_start_time = now

                # Check if silence duration exceeded threshold
                silence_duration_ms = (now - self._silence_start_time) * 1000.0
                if silence_duration_ms >= self.config.vad.min_silence_duration_ms:
                    # User completed their utterance
                    self._is_speaking_user = False
                    self._silence_start_time = None

                    # Trim trailing silence chunks, preserving only speech_pad_ms of padding
                    pad_chunks = max(1, int(np.ceil(self.config.vad.speech_pad_ms / self.config.audio.frame_size_ms)))
                    silence_chunks_count = max(0, int(np.round(silence_duration_ms / self.config.audio.frame_size_ms)))
                    excess_chunks = max(0, silence_chunks_count - pad_chunks)

                    if excess_chunks > 0 and len(self._speech_chunks) > excess_chunks:
                        audio_chunks_to_stt = self._speech_chunks[:-excess_chunks]
                    else:
                        audio_chunks_to_stt = self._speech_chunks

                    if audio_chunks_to_stt:
                        full_audio = np.concatenate(audio_chunks_to_stt)
                    else:
                        full_audio = np.array([], dtype=np.float32)

                    self._speech_chunks.clear()
                    self._pre_speech_buffer.clear()
                    self.vad.reset_states()

                    # Schedule async processing in event loop
                    if full_audio.size > 0:
                        self._schedule_utterance_processing(full_audio)
            else:
                # Passive silence: keep ring buffer fresh
                self._pre_speech_buffer.append(chunk)

    def _save_debug_record(self, audio_data: np.ndarray, text: str, metrics: Dict[str, Any]) -> None:
        """Dump audio waveform to WAV and metadata/metrics to JSON for forensic debugging."""
        if not self.config.debug_audio_dir:
            return
        try:
            debug_dir = Path(self.config.debug_audio_dir).expanduser().resolve()
            debug_dir.mkdir(parents=True, exist_ok=True)
            self._debug_record_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_name = f"utterance_{timestamp}_{self._debug_record_count:04d}"

            audio_flat = np.nan_to_num(np.asarray(audio_data, dtype=np.float32).ravel(), nan=0.0, posinf=0.0, neginf=0.0)

            wav_path = debug_dir / f"{base_name}.wav"
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.config.audio.input_sample_rate)
                pcm = (np.clip(audio_flat, -1.0, 1.0) * 32767).astype(np.int16)
                wf.writeframes(pcm.tobytes())

            json_path = debug_dir / f"{base_name}.json"
            record = {
                "timestamp": timestamp,
                "duration_seconds": float(audio_flat.size / self.config.audio.input_sample_rate),
                "sample_rate": self.config.audio.input_sample_rate,
                "transcription": text,
                "metrics": _sanitize_for_json(metrics),
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        except Exception:
            # Defensive logging: debugging file errors must never crash the voice pipeline
            pass

    async def _handle_user_utterance(self, audio_data: np.ndarray) -> None:
        """Process completed user audio utterance through STT -> LLM -> TTS."""
        audio_arr = np.nan_to_num(np.asarray(audio_data, dtype=np.float32).ravel(), nan=0.0, posinf=0.0, neginf=0.0)

        if audio_arr.size < (self.config.audio.input_sample_rate * 0.4):
            # Ignore audio bursts shorter than 400ms (clicks, bumps, breath)
            if not self._is_speaking_user:
                self.on_status("IDLE", "Listening...")
            return

        # Compute RMS energy and drop low-energy room noise / sighs
        rms = float(np.sqrt(np.mean(audio_arr ** 2))) if audio_arr.size > 0 else 0.0
        if rms < self.config.vad.min_rms_threshold:
            if self.config.debug_audio_dir:
                self._save_debug_record(
                    audio_data=audio_arr,
                    text="",
                    metrics={
                        "status": "dropped_low_rms",
                        "rms": rms,
                        "min_rms_threshold": self.config.vad.min_rms_threshold,
                    },
                )
            if not self._is_speaking_user:
                self.on_status("IDLE", "Listening...")
            return

        async with self._utterance_lock:
            self.on_status("TRANSCRIBING", "Running faster-whisper on CPU...")
            stt_start = time.perf_counter()

            # Run STT in thread pool to prevent blocking the async loop
            loop = asyncio.get_running_loop()
            user_text = await loop.run_in_executor(None, self.stt.transcribe, audio_arr)
            stt_duration = (time.perf_counter() - stt_start) * 1000.0

            if self.config.debug_audio_dir:
                stt_metrics = getattr(self.stt, "last_metrics", {})
                metrics = dict(stt_metrics) if isinstance(stt_metrics, dict) else {}
                metrics["stt_duration_ms"] = stt_duration
                metrics["rms"] = rms
                self._save_debug_record(
                    audio_data=audio_arr,
                    text=user_text or "",
                    metrics=metrics,
                )

            if not user_text:
                self.on_status("IDLE", "Silence or unintelligible speech detected.")
                return

            self.on_transcription(f"{user_text} (STT: {stt_duration:.1f}ms)")
            self._conversation_history.append({"role": "user", "content": user_text})

            # Cap history to last 10 messages for memory efficiency
            if len(self._conversation_history) > 10:
                self._conversation_history = self._conversation_history[-10:]

            self.on_status("THINKING", "Streaming LLM response...")
            assistant_reply_full: List[str] = []

            try:
                async for sentence in self.llm.generate_sentences(
                    user_message=user_text,
                    conversation_history=self._conversation_history[:-1]
                ):
                    if not sentence:
                        continue

                    assistant_reply_full.append(sentence)
                    self.on_response(sentence)

                    # Synthesize and enqueue sentence audio immediately
                    self.on_status("SPEAKING", f"Synthesizing: '{sentence}'")
                    tts_start = time.perf_counter()
                    audio_samples, sr = await loop.run_in_executor(
                        None, self.tts.synthesize, sentence
                    )
                    tts_duration = (time.perf_counter() - tts_start) * 1000.0

                    if audio_samples.size > 0:
                        self.audio.queue_audio(audio_samples)

                full_reply_text = " ".join(assistant_reply_full)
                if full_reply_text:
                    self._conversation_history.append(
                        {"role": "assistant", "content": full_reply_text}
                    )

            except asyncio.CancelledError:
                if assistant_reply_full:
                    partial_reply = " ".join(assistant_reply_full).strip()
                    if partial_reply:
                        self._conversation_history.append(
                            {"role": "assistant", "content": partial_reply}
                        )
                self.on_status("INTERRUPTED", "Generation cancelled due to barge-in.")
                raise
            except Exception as e:
                self.on_status("ERROR", f"Pipeline error: {e}")
            finally:
                if not self._is_speaking_user:
                    self.on_status("IDLE", "Listening...")

    async def run(self) -> None:
        """Start the live voice interaction loop."""
        self._event_loop = asyncio.get_running_loop()
        self._running = True

        self.audio.start_output()
        self.audio.start_input(self._audio_callback, on_level=self.on_level)

        self.on_status("READY", "Sonus voice pipeline active. Speak into your microphone.")

        try:
            while self._running:
                await asyncio.sleep(0.1)
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop streams and release hardware resources."""
        self._running = False
        self._cancel_processing_task()
        self.audio.close()
        self.on_status("STOPPED", "Audio hardware released.")
