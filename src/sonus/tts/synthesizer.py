"""
High-fidelity Text-to-Speech synthesis using Kokoro-82M ONNX on CPU.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
from kokoro_onnx import Kokoro

from sonus.config import TTSConfig


class KokoroSynthesizer:
    """Fast neural voice synthesizer generating 24kHz studio audio on CPU."""

    def __init__(self, config: TTSConfig):
        self.config = config
        model_path = Path(config.model_path)
        voices_path = Path(config.voices_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Kokoro ONNX model not found at {model_path}")
        if not voices_path.exists():
            raise FileNotFoundError(f"Voices file not found at {voices_path}")

        self._kokoro = Kokoro(
            model_path=str(model_path),
            voices_path=str(voices_path),
        )

    def synthesize(self, text: str, voice: str = "") -> Tuple[np.ndarray, int]:
        """
        Synthesize text into raw 24kHz float32 audio.
        
        Args:
            text: Sentence to speak.
            voice: Voice identifier (defaults to config.voice).
            
        Returns:
            Tuple[np.ndarray, int]: (audio_samples, sample_rate)
        """
        clean_text = text.strip()
        if not clean_text:
            return np.array([], dtype=np.float32), 24000

        target_voice = voice or self.config.voice
        samples, sample_rate = self._kokoro.create(
            text=clean_text,
            voice=target_voice,
            speed=self.config.speed,
            lang="en-us",
        )
        return samples, sample_rate

    def get_available_voices(self) -> List[str]:
        """Return list of voice profiles available in voices.json."""
        return self._kokoro.get_voices()
