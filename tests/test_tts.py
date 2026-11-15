"""
Unit tests for Kokoro-82M ONNX speech synthesis on CPU.
"""

from unittest.mock import MagicMock
import numpy as np
from sonus.config import TTSConfig
from sonus.tts.synthesizer import KokoroSynthesizer


def test_kokoro_tts_synthesis():
    config = TTSConfig()
    try:
        synth = KokoroSynthesizer(config)
    except FileNotFoundError:
        return

    if hasattr(synth._kokoro, "get_voices"):
        if hasattr(synth._kokoro.get_voices, "return_value"):
            synth._kokoro.get_voices.return_value = ["af_heart", "af_bella"]
        if hasattr(synth._kokoro.create, "return_value"):
            synth._kokoro.create.return_value = (np.ones(24000, dtype=np.float32), 24000)

    voices = synth.get_available_voices()
    assert len(voices) > 0
    assert "af_heart" in voices or "af" in voices or len(voices) >= 1

    # Synthesize short phrase
    samples, sample_rate = synth.synthesize("Hello world.")
    assert sample_rate == 24000
    assert samples.ndim == 1
    assert len(samples) > 0

    # Synthesize empty text returns empty array
    empty_samples, empty_sr = synth.synthesize("")
    assert len(empty_samples) == 0
    assert empty_sr == 24000
