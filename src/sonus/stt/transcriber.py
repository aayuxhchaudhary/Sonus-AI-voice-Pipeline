"""
Speech-to-Text inference engine powered by faster-whisper (CTranslate2 INT8).
"""

from typing import Optional
import numpy as np
from faster_whisper import WhisperModel

from sonus.config import STTConfig


class WhisperTranscriber:
    """High-speed local speech transcriber optimized for AVX-512 CPU."""

    def __init__(self, config: STTConfig):
        self.config = config
        self._model = WhisperModel(
            model_size_or_path=config.model_size,
            device=config.device,
            compute_type=config.compute_type,
            cpu_threads=config.num_threads,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe raw 16kHz float32 audio waveform.
        
        Args:
            audio: 1D NumPy float32 array normalized between -1.0 and 1.0.
            sample_rate: Audio sampling frequency (Whisper expects 16kHz).
            
        Returns:
            Transcribed text string.
        """
        if audio.size == 0:
            return ""

        # faster-whisper accepts float32 numpy arrays directly
        segments, info = self._model.transcribe(
            audio,
            beam_size=1,            # Greedy decoding for lowest latency
            language="en",
            vad_filter=False,       # Upstream Silero VAD already sliced the audio
            temperature=0.0,
        )

        texts = [segment.text.strip() for segment in segments]
        transcription = " ".join(t for t in texts if t)
        return transcription.strip()
