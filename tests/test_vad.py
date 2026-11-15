"""
Unit tests for Silero VAD v5 stateful inference and 64-sample history context.
"""

import wave
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from sonus.config import VADConfig, TTSConfig
from sonus.vad.silero import SileroVAD


def load_speech_fixture() -> np.ndarray:
    """Load authentic 16kHz audio sample from fixtures, or synthesize fallback."""
    fixture_path = Path(__file__).parent / "fixtures" / "speech_sample_16k.wav"
    if fixture_path.exists():
        with wave.open(str(fixture_path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(frames, dtype=np.int16)
            return (pcm.astype(np.float32) / 32768.0)

    try:
        from sonus.tts.synthesizer import KokoroSynthesizer
        tts = KokoroSynthesizer(TTSConfig())
        audio_24k, sr = tts.synthesize("Hello world, this is a test speech sample.")
        num_target = int(len(audio_24k) * 16000 / 24000)
        return np.interp(
            np.linspace(0, len(audio_24k), num_target, endpoint=False),
            np.arange(len(audio_24k)),
            audio_24k
        ).astype(np.float32)
    except Exception:
        t = np.linspace(0, 1.5, 24000, endpoint=False, dtype=np.float32)
        return (
            0.3 * np.sin(2 * np.pi * 130 * t) +
            0.4 * np.sin(2 * np.pi * 730 * t) +
            0.2 * np.sin(2 * np.pi * 1090 * t)
        ).astype(np.float32)


def test_silero_vad_synthetic_silence():
    config = VADConfig()
    vad = SileroVAD(config, sample_rate=16000)

    if hasattr(vad._session, "_mock_name") or type(vad._session).__name__ in ("MagicMock", "Mock"):
        vad._session.run = MagicMock(return_value=(
            np.array([[[0.05]]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32)
        ))

    # 512 samples of digital silence (32ms at 16kHz)
    silence = np.zeros(512, dtype=np.float32)
    is_speech, prob = vad.is_speech(silence)

    assert not is_speech
    assert 0.0 <= prob <= 0.2


def test_silero_vad_reset_state():
    config = VADConfig()
    vad = SileroVAD(config, sample_rate=16000)

    if hasattr(vad._session, "_mock_name") or type(vad._session).__name__ in ("MagicMock", "Mock"):
        vad._session.run = MagicMock(return_value=(
            np.array([[[0.5]]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32)
        ))

    # Corrupt/populate context and state
    test_chunk = np.ones(512, dtype=np.float32) * 0.5
    vad.is_speech(test_chunk)
    assert not np.all(vad._context == 0.0)

    vad.reset_states()
    assert vad._state.shape == (2, 1, 128)
    assert np.all(vad._state == 0.0)
    assert vad._context.shape == (64,)
    assert np.all(vad._context == 0.0)


def test_silero_vad_context_prepended():
    config = VADConfig()
    vad = SileroVAD(config, sample_rate=16000)

    inputs_passed = []
    if hasattr(vad._session, "_mock_name") or type(vad._session).__name__ in ("MagicMock", "Mock"):
        def mock_run(out_names, ort_inputs):
            inputs_passed.append(ort_inputs["input"])
            return (
                np.array([[[0.1]]], dtype=np.float32),
                np.zeros((2, 1, 128), dtype=np.float32)
            )
        vad._session.run = mock_run

    # Create distinct chunk where the last 64 samples are identifiable
    frame1 = np.zeros(512, dtype=np.float32)
    frame1[-64:] = np.linspace(0.1, 0.8, 64, dtype=np.float32)

    vad.is_speech(frame1)

    # Verify that vad._context now holds the last 64 samples of frame1
    assert vad._context.shape == (64,)
    assert np.allclose(vad._context, frame1[-64:])

    # Frame 2 with zeros
    frame2 = np.zeros(512, dtype=np.float32)
    vad.is_speech(frame2)
    # Now context should be all zeros from the end of frame2
    assert np.allclose(vad._context, np.zeros(64, dtype=np.float32))

    if inputs_passed:
        # Silero VAD v5 requires [1, 576] input shape (64 context + 512 frame)
        assert all(inp.shape == (1, 576) for inp in inputs_passed)
        # Context from frame 1 must be prepended to frame 2
        assert np.allclose(inputs_passed[1][0, :64], frame1[-64:])


def test_silero_vad_real_speech_samples():
    """Test Silero VAD with authentic synthesized speech audio samples."""
    config = VADConfig()
    vad = SileroVAD(config, sample_rate=16000)
    vad.reset_states()

    if hasattr(vad._session, "_mock_name") or type(vad._session).__name__ in ("MagicMock", "Mock"):
        def mock_run(out_names, ort_inputs):
            inp = ort_inputs["input"]
            assert inp.shape == (1, 576)
            rms = float(np.sqrt(np.mean(inp**2)))
            prob = min(0.99, max(0.01, rms * 4.0))
            return (
                np.array([[[prob]]], dtype=np.float32),
                np.zeros((2, 1, 128), dtype=np.float32)
            )
        vad._session.run = mock_run

    speech_16k = load_speech_fixture()

    detected_speech_chunks = 0
    max_prob = 0.0

    # Feed sequential 512-sample chunks
    for i in range(0, len(speech_16k) - 512, 512):
        chunk = speech_16k[i:i + 512]
        is_speech, prob = vad.is_speech(chunk)
        if prob > max_prob:
            max_prob = prob
        if is_speech:
            detected_speech_chunks += 1

    # Real speech audio must trigger positive speech detection
    assert detected_speech_chunks > 0
    assert max_prob >= config.threshold
