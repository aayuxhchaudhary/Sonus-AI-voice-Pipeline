"""
Unit tests for Silero VAD stateful inference.
"""

import numpy as np
from sonus.config import VADConfig
from sonus.vad.silero import SileroVAD


def test_silero_vad_synthetic_silence():
    config = VADConfig()
    vad = SileroVAD(config, sample_rate=16000)

    # 512 samples of digital silence (32ms at 16kHz)
    silence = np.zeros(512, dtype=np.float32)
    is_speech, prob = vad.is_speech(silence)

    assert not is_speech
    assert 0.0 <= prob <= 0.2


def test_silero_vad_reset_state():
    config = VADConfig()
    vad = SileroVAD(config, sample_rate=16000)
    vad.reset_states()
    assert vad._state.shape == (2, 1, 128)
