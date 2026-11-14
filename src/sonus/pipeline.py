"""
Core real-time event-driven conversational pipeline orchestrator.
"""

import asyncio
import time
from typing import Callable, List, Optional
import numpy as np

from sonus.config import SonusConfig
from sonus.audio.stream import AudioStreamManager
from sonus.audio.ring_buffer import AudioRingBuffer
from sonus.vad.silero import SileroVAD
from sonus.stt.transcriber import WhisperTranscriber
from sonus.llm.client import StreamingLLMClient
from sonus.tts.synthesizer import KokoroSynthesizer


class VoicePipeline:
    """End-to-end voice-to-voice conversational orchestrator with barge-in support."""

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
        self._conversation_history: List[dict] = []

    def _audio_callback(self, chunk: np.ndarray) -> None:
        """Invoked on audio thread for every incoming 32ms microphone frame."""
        if not self._running or self._event_loop is None:
            return

        is_speech, score = self.vad.is_speech(chunk)

        # 1. Barge-in detection: User speaks while the assistant is talking
        if is_speech and self.audio.is_playing:
            self.on_status("BARGE_IN", "User interruption detected; interrupting playback.")
            self.audio.stop_playback()
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
            self._is_speaking_user = True
            self._speech_chunks = [chunk]
            self.vad.reset_states()
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
                    full_audio = np.concatenate(self._speech_chunks)
                    self._speech_chunks.clear()
                    self._pre_speech_buffer.clear()
                    self.vad.reset_states()

                    # Schedule async processing in event loop
                    asyncio.run_coroutine_threadsafe(
                        self._handle_user_utterance(full_audio),
                        self._event_loop
                    )
            else:
                # Passive silence: keep ring buffer fresh
                self._pre_speech_buffer.append(chunk)

    async def _handle_user_utterance(self, audio_data: np.ndarray) -> None:
        """Process completed user audio utterance through STT -> LLM -> TTS."""
        if audio_data.size < (self.config.audio.input_sample_rate * 0.4):
            # Ignore audio bursts shorter than 400ms (clicks, bumps, breath)
            return

        self.on_status("TRANSCRIBING", "Running faster-whisper on CPU...")
        stt_start = time.perf_counter()

        # Run STT in thread pool to prevent blocking the async loop
        loop = asyncio.get_running_loop()
        user_text = await loop.run_in_executor(None, self.stt.transcribe, audio_data)
        stt_duration = (time.perf_counter() - stt_start) * 1000.0

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
            self.on_status("INTERRUPTED", "Generation cancelled due to barge-in.")
        except Exception as e:
            self.on_status("ERROR", f"Pipeline error: {e}")
        finally:
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
        self.audio.close()
        self.on_status("STOPPED", "Audio hardware released.")
