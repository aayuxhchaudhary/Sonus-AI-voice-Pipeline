"""
Real-time audio I/O subsystem powered by sounddevice / PortAudio.

Handles hardware sample-rate auto-negotiation, linear interpolation resampling,
and zero-latency preemption for conversational barge-in interruptions.
"""

import queue
import threading
from typing import Callable, Optional, Tuple, Any
import numpy as np
import sounddevice as sd

from sonus.config import AudioConfig


class AudioStreamManager:
    """
    Coordinates asynchronous monophonic microphone capture and preemptible speaker output.
    
    Concurrency Model:
        - Input Stream: PortAudio invokes an asynchronous C-level real-time audio callback.
          Captured frames are conditioned (resampled, shape-clamped, RMS-metered) and
          forwarded to the pipeline callback.
        - Output Stream: Runs in a dedicated background worker thread (`_playback_thread`).
          Synthesized sentence audio is queued via `_playback_queue` and written in 1024-sample
          slices. Atomic events (`_interrupt_playback`, `_is_playing`) provide instantaneous
          (<25ms) audio truncation upon user barge-in.
    """

    def __init__(self, config: AudioConfig):
        self.config = config
        self._input_stream: Optional[sd.InputStream] = None
        self._output_stream: Optional[sd.OutputStream] = None

        self._playback_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._playback_thread: Optional[threading.Thread] = None
        self._is_playing: threading.Event = threading.Event()
        self._stop_playback_flag: threading.Event = threading.Event()
        self._interrupt_playback: threading.Event = threading.Event()

        self._active_input_device: Optional[int] = None
        self._input_decimation_ratio: int = 1

    def _resolve_input_device_and_rate(self) -> Tuple[Any, int, int]:
        """
        Probe and negotiate viable hardware input endpoints and native clock frequencies.
        
        Attempts direct negotiation at 16,000 Hz (standardized for Silero VAD and Whisper STT).
        If the endpoint requires a hardware-locked clock (e.g. 48,000 Hz or 44,100 Hz), calculates
        the integer decimation factor or initializes linear interpolation resampling.

        Returns:
            Tuple[device_id, hardware_sample_rate, decimation_ratio]
        """
        if self.config.input_device is not None:
            candidates = [self.config.input_device]
        else:
            candidates = [None]
            try:
                devices = sd.query_devices()
                for idx, d in enumerate(devices):
                    if d.get("max_input_channels", 0) > 0 and idx not in candidates:
                        candidates.append(idx)
            except Exception:
                pass

        for dev in candidates:
            # Stage 1: Attempt standard 16 kHz capture
            try:
                sd.check_input_settings(
                    device=dev,
                    channels=self.config.channels,
                    dtype="float32",
                    samplerate=self.config.input_sample_rate
                )
                return dev, self.config.input_sample_rate, 1
            except Exception:
                pass

            # Stage 2: Probe native hardware rate and downsample
            try:
                info = sd.query_devices(dev) if dev is not None else sd.query_devices(kind="input")
                native_sr = int(info.get("default_samplerate", 48000))
                sd.check_input_settings(
                    device=dev,
                    channels=self.config.channels,
                    dtype="float32",
                    samplerate=native_sr
                )
                ratio = max(1, native_sr // self.config.input_sample_rate)
                return dev, native_sr, ratio
            except Exception:
                continue

        return None, self.config.input_sample_rate, 1

    def start_input(
        self,
        callback: Callable[[np.ndarray], None],
        on_level: Optional[Callable[[float], None]] = None
    ) -> None:
        """
        Initialize and activate the real-time PortAudio microphone stream.
        
        Args:
            callback: Invoked on every completed 32ms (512-sample) audio frame.
            on_level: Optional metering hook receiving frame RMS energy for VU visualization.
        """
        dev, hw_sr, decimation = self._resolve_input_device_and_rate()
        self._active_input_device = dev
        self._input_decimation_ratio = decimation

        target_sr = self.config.input_sample_rate
        target_frame_samples = int(target_sr * (self.config.frame_size_ms / 1000.0))
        hw_blocksize = int(round(target_frame_samples * hw_sr / target_sr))

        def sd_callback(indata, frames, time_info, status):
            if status:
                pass

            mono_raw = indata[:, 0].astype(np.float32)

            # Sample rate alignment: direct stride decimation or linear interpolation
            if hw_sr != target_sr:
                if decimation > 1 and hw_sr % target_sr == 0:
                    mono_chunk = mono_raw[::decimation]
                else:
                    mono_chunk = np.interp(
                        np.linspace(0, len(mono_raw), target_frame_samples, endpoint=False),
                        np.arange(len(mono_raw)),
                        mono_raw
                    ).astype(np.float32)
            else:
                mono_chunk = mono_raw

            # Frame size validation: enforce exact 512-sample invariant for VAD
            if len(mono_chunk) != target_frame_samples:
                if len(mono_chunk) > target_frame_samples:
                    mono_chunk = mono_chunk[:target_frame_samples]
                else:
                    mono_chunk = np.pad(mono_chunk, (0, target_frame_samples - len(mono_chunk)))

            if on_level:
                rms = float(np.sqrt(np.mean(mono_chunk**2)))
                on_level(rms)

            callback(mono_chunk)

        self._input_stream = sd.InputStream(
            device=dev,
            samplerate=hw_sr,
            channels=self.config.channels,
            dtype="float32",
            blocksize=hw_blocksize,
            callback=sd_callback,
        )
        self._input_stream.start()

    @property
    def active_input_device(self) -> Optional[int]:
        """Return the device index currently capturing audio."""
        return self._active_input_device

    def start_output(self) -> None:
        """Start background audio playback worker thread."""
        self._stop_playback_flag.clear()
        self._interrupt_playback.clear()
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()

    def _playback_worker(self) -> None:
        """Worker thread that streams generated audio chunks to the speakers."""
        try:
            with sd.OutputStream(
                device=self.config.output_device,
                samplerate=self.config.output_sample_rate,
                channels=self.config.channels,
                dtype="float32",
            ) as stream:
                self._output_stream = stream
                while not self._stop_playback_flag.is_set():
                    try:
                        audio_chunk = self._playback_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if audio_chunk is None:
                        try:
                            self._playback_queue.task_done()
                        except ValueError:
                            pass
                        self._is_playing.clear()
                        continue

                    if self._interrupt_playback.is_set():
                        try:
                            self._playback_queue.task_done()
                        except ValueError:
                            pass
                        if self._playback_queue.empty():
                            self._is_playing.clear()
                        continue

                    self._is_playing.set()
                    chunk_slice_size = 1024
                    offset = 0
                    while (
                        offset < len(audio_chunk)
                        and not self._stop_playback_flag.is_set()
                        and not self._interrupt_playback.is_set()
                    ):
                        end = min(offset + chunk_slice_size, len(audio_chunk))
                        slice_data = audio_chunk[offset:end]
                        try:
                            stream.write(slice_data)
                        except Exception:
                            break
                        offset = end

                    try:
                        self._playback_queue.task_done()
                    except ValueError:
                        pass
                    if self._playback_queue.empty() or self._interrupt_playback.is_set():
                        self._is_playing.clear()
        except Exception:
            pass
        finally:
            self._output_stream = None
            self._is_playing.clear()

    def queue_audio(self, audio_data: np.ndarray) -> None:
        """Enqueue audio segment for playback."""
        if audio_data.size > 0:
            self._interrupt_playback.clear()
            self._playback_queue.put(audio_data)

    def stop_playback(self) -> None:
        """Immediate barge-in interruption: drop playback queue and silence active audio."""
        self._interrupt_playback.set()
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
                self._playback_queue.task_done()
            except (queue.Empty, ValueError):
                break
        self._is_playing.clear()

    @property
    def is_playing(self) -> bool:
        """Check if speech audio is currently playing through speakers."""
        return self._is_playing.is_set()

    def close(self) -> None:
        """Clean up streams and worker threads."""
        self.stop_playback()
        self._stop_playback_flag.set()
        if self._input_stream:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except Exception:
                pass
            self._input_stream = None
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)
            self._playback_thread = None
