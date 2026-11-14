"""
Unit tests for Kokoro-82M ONNX speech synthesis on CPU.
"""

from sonus.config import TTSConfig
from sonus.tts.synthesizer import KokoroSynthesizer


def test_kokoro_tts_synthesis():
    config = TTSConfig()
    synth = KokoroSynthesizer(config)

    voices = synth.get_available_voices()
    assert len(voices) > 0
    assert "af_heart" in voices or "af" in voices or len(voices) >= 1

    # Synthesize short phrase
    samples, sample_rate = synth.synthesize("Hello world.")
    assert sample_rate == 24000
    assert samples.ndim == 1
    assert len(samples) > 0
