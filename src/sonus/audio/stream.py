"""
Low-latency audio input capture and interruptible playback engine using sounddevice.
Auto-negotiates hardware sample rates and audio device selection on Linux/ALSA.
"""

import queue
import threading
from typing import Callable, Optional, Tuple, Any
import numpy as np
import sounddevice as sd

from sonus.config import AudioConfig


class AudioStreamManager:
    """Manages simultaneous microphone input and speaker output streams."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self._input_stream: Optional[sd.InputStream] = None
        self._output_stream: Optional[sd.OutputStream] = None

        self._playback_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._playback_thread: Optional[threading.Thread] = None
        self._is_playing: threading.Event = threading.Event()
        self._stop_playback_flag: threading.Event = threading.Event()

        self._active_input_device: Optional[int] = None
        self._input_decimation_ratio: int = 1

    def _resolve_input_device_and_rate(self) -> Tuple[Any, int, int]:
        """
        Determine the optimal hardware input device and sampling rate.
        Returns: (device_id, hardware_sample_rate, decimation_ratio)
        """
        if self.config.input_device is not None:
            candidates = [self.config.input_device]
        else:
            # Query all devices and prioritize the real hardware analog mic (hw:0,0 / ALC3254)
            candidates = []
            devices = sd.query_devices()
            for idx, d in enumerate(devices):
                if d.get("max_input_channels", 0) > 0:
                    name = d.get("name", "").lower()
                    if "alc" in name or "hw:0,0" in name:
                        candidates.insert(0, idx)
                    elif "built-in" in name or "analog" in name:
                        candidates.append(idx)
                    else:
                        candidates.append(idx)

            if not candidates:
                candidates = [None]  # Fall back to PortAudio default

        for dev in candidates:
            # First try requesting 16000Hz directly
            try:
                sd.check_input_settings(device=dev, channels=self.config.channels, dtype="float32", samplerate=16000)
                return dev, 16000, 1
            except Exception:
                pass

            # Try the device native sample rate (e.g. 48000Hz) and downsample
            try:
                info = sd.query_devices(dev)
                native_sr = int(info.get("default_samplerate", 48000))
                sd.check_input_settings(device=dev, channels=self.config.channels, dtype="float32", samplerate=native_sr)
                # Compute integer decimation ratio (e.g. 48000 // 16000 = 3)
                ratio = max(1, native_sr // self.config.input_sample_rate)
                return dev, native_sr, ratio
            except Exception:
                continue

        # Ultimate fallback
        return None, 16000, 1

    def start_input(
        self,
        callback: Callable[[np.ndarray], None],
        on_level: Optional[Callable[[float], None]] = None
    ) -> None:
        """Start microphone capture stream with auto-negotiated device and downsampling."""
        dev, hw_sr, decimation = self._resolve_input_device_and_rate()
        self._active_input_device = dev
        self._input_decimation_ratio = decimation

        target_sr = self.config.input_sample_rate
        # Target frame is 512 samples at 16kHz (32ms)
        target_frame_samples = int(target_sr * (self.config.frame_size_ms / 1000.0))
        # Blocksize on hardware device
        hw_blocksize = target_frame_samples * decimation

        def sd_callback(indata, frames, time_info, status):
            if status:
                pass

            mono_raw = indata[:, 0].astype(np.float32)

            # Downsample if hardware runs at 48kHz
            if decimation > 1:
                mono_chunk = mono_raw[::decimation]
            else:
                mono_chunk = mono_raw

            # Trim or pad to exact frame size
            if len(mono_chunk) != target_frame_samples:
                if len(mono_chunk) > target_frame_samples:
                    mono_chunk = mono_chunk[:target_frame_samples]
                else:
                    mono_chunk = np.pad(mono_chunk, (0, target_frame_samples - len(mono_chunk)))

            if on_level:
                peak = float(np.max(np.abs(mono_chunk)))
                on_level(peak)

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
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()

    def _playback_worker(self) -> None:
        """Worker thread that streams generated audio chunks to the speakers."""
        with sd.OutputStream(
            device=self.config.output_device,
            samplerate=self.config.output_sample_rate,
            channels=self.config.channels,
            dtype="float32",
        ) as stream:
            while not self._stop_playback_flag.is_set():
                try:
                    audio_chunk = self._playback_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if audio_chunk is None:
                    self._is_playing.clear()
                    continue

                self._is_playing.set()
                chunk_slice_size = 1024
                offset = 0
                while offset < len(audio_chunk) and not self._stop_playback_flag.is_set():
                    end = min(offset + chunk_slice_size, len(audio_chunk))
                    slice_data = audio_chunk[offset:end]
                    stream.write(slice_data)
                    offset = end

                self._playback_queue.task_done()
                if self._playback_queue.empty():
                    self._is_playing.clear()

    def queue_audio(self, audio_data: np.ndarray) -> None:
        """Enqueue audio segment for playback."""
        if audio_data.size > 0:
            self._playback_queue.put(audio_data)

    def stop_playback(self) -> None:
        """Immediate barge-in interruption: drop playback queue and silence active audio."""
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
            self._input_stream.stop()
            self._input_stream.close()
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)
