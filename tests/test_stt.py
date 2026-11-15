"""
Unit tests for faster-whisper INT8 CPU transcription and hallucination filtering.
"""

import asyncio
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

try:
    import pytest
except ImportError:
    class _MockPytest:
        class mark:
            anyio = lambda fn: fn
    pytest = _MockPytest()  # type: ignore

from sonus.config import SonusConfig, STTConfig
from sonus.pipeline import VoicePipeline
from sonus.stt.transcriber import WhisperTranscriber


def load_speech_fixture() -> np.ndarray:
    """Load authentic 16kHz audio sample from fixtures, or return synthetic fallback."""
    fixture_path = Path(__file__).parent / "fixtures" / "speech_sample_16k.wav"
    if fixture_path.exists():
        with wave.open(str(fixture_path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(frames, dtype=np.int16)
            return (pcm.astype(np.float32) / 32768.0)
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


def test_whisper_transcriber_silence():
    """Verify silence produces empty transcription without hallucinations."""
    config = STTConfig(model_size="tiny.en")
    transcriber = WhisperTranscriber(config)

    if hasattr(transcriber._model.transcribe, "return_value"):
        mock_seg = MagicMock()
        mock_seg.text = "You"
        mock_seg.no_speech_prob = 0.95
        transcriber._model.transcribe.return_value = ([mock_seg], None)

    # 1 second of silence at 16kHz
    silence = np.zeros(16000, dtype=np.float32)
    result = transcriber.transcribe(silence)
    assert isinstance(result, str)
    # Silence hallucination filter should yield empty string
    assert result == ""


def test_whisper_no_speech_prob_filtering():
    """Verify segments with high no_speech_prob are filtered out as hallucinations."""
    config = STTConfig(model_size="tiny.en")
    transcriber = WhisperTranscriber(config)

    # Mock segment with high no_speech_prob (hallucination)
    mock_hallucination = MagicMock()
    mock_hallucination.text = " Thank you for watching. "
    mock_hallucination.no_speech_prob = 0.85

    # Mock segment with real speech (low no_speech_prob)
    mock_real = MagicMock()
    mock_real.text = " Hello world. "
    mock_real.no_speech_prob = 0.05

    transcriber._model.transcribe = MagicMock(
        return_value=([mock_hallucination, mock_real], None)
    )

    audio = np.zeros(16000, dtype=np.float32)
    result = transcriber.transcribe(audio)

    assert "Thank you for watching" not in result
    assert "Hello world." in result


def test_whisper_transcriber_real_speech():
    """Verify transcription on authentic speech audio samples."""
    config = STTConfig(model_size="tiny.en")
    transcriber = WhisperTranscriber(config)
    speech_16k = load_speech_fixture()

    if hasattr(transcriber._model.transcribe, "side_effect") or hasattr(transcriber._model.transcribe, "return_value"):
        mock_seg = MagicMock()
        mock_seg.text = "Hello world this is a real speech test."
        mock_seg.no_speech_prob = 0.02
        transcriber._model.transcribe = MagicMock(return_value=([mock_seg], None))

    result = transcriber.transcribe(speech_16k)
    assert isinstance(result, str)
    assert "hello" in result.lower() or "speech" in result.lower() or "world" in result.lower()


def test_whisper_initial_prompt_and_hotwords():
    """Verify STTConfig and WhisperTranscriber configure initial_prompt and hotwords."""
    config = STTConfig()
    assert "Aayush" in config.initial_prompt
    assert "JSON" in config.initial_prompt
    assert "Python" in config.initial_prompt
    assert "Aayush" in config.hotwords
    assert "JSON" in config.hotwords
    assert config.condition_on_previous_text is False
    assert config.repetition_penalty == 1.2
    assert config.no_repeat_ngram_size == 3
    assert config.compression_ratio_threshold == 2.4
    assert config.log_prob_threshold == -1.0
    assert config.no_speech_threshold == 0.6

    transcriber = WhisperTranscriber(config)
    mock_seg = MagicMock()
    mock_seg.text = "Testing prompt"
    mock_seg.no_speech_prob = 0.01
    mock_seg.compression_ratio = 1.1
    mock_seg.avg_logprob = -0.2

    transcriber._model.transcribe = MagicMock(return_value=([mock_seg], None))

    audio = np.ones(16000, dtype=np.float32) * 0.1
    result = transcriber.transcribe(audio)
    assert result == "Testing prompt"

    call_kwargs = transcriber._model.transcribe.call_args[1]
    assert call_kwargs["initial_prompt"] == config.initial_prompt
    assert call_kwargs["hotwords"] == config.hotwords
    assert call_kwargs["condition_on_previous_text"] is False
    assert call_kwargs["repetition_penalty"] == 1.2
    assert call_kwargs["no_repeat_ngram_size"] == 3
    assert call_kwargs["vad_filter"] is True
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["beam_size"] == 1
    assert call_kwargs["compression_ratio_threshold"] == 2.4
    assert call_kwargs["log_prob_threshold"] == -1.0
    assert call_kwargs["no_speech_threshold"] == 0.6


def test_whisper_compression_ratio_filtering():
    """Verify segments with compression_ratio > compression_ratio_threshold are filtered as hallucinations."""
    config = STTConfig(compression_ratio_threshold=2.4)
    transcriber = WhisperTranscriber(config)

    mock_repetitive = MagicMock()
    mock_repetitive.text = "json json json json json singing singing"
    mock_repetitive.no_speech_prob = 0.05
    mock_repetitive.compression_ratio = 2.8  # Exceeds 2.4 threshold
    mock_repetitive.avg_logprob = -0.3

    mock_valid = MagicMock()
    mock_valid.text = "how do I parse a json file"
    mock_valid.no_speech_prob = 0.05
    mock_valid.compression_ratio = 1.3  # Below 2.4 threshold
    mock_valid.avg_logprob = -0.3

    transcriber._model.transcribe = MagicMock(
        return_value=([mock_repetitive, mock_valid], None)
    )

    audio = np.ones(16000, dtype=np.float32) * 0.1
    result = transcriber.transcribe(audio)

    assert "singing" not in result
    assert "how do I parse a json file" in result


def test_whisper_log_prob_filtering():
    """Verify segments with avg_logprob < log_prob_threshold are filtered out."""
    config = STTConfig(log_prob_threshold=-1.0)
    transcriber = WhisperTranscriber(config)

    mock_low_confidence = MagicMock()
    mock_low_confidence.text = "mumbled unsaid gibberish"
    mock_low_confidence.no_speech_prob = 0.1
    mock_low_confidence.compression_ratio = 1.2
    mock_low_confidence.avg_logprob = -1.8  # Worse than -1.0 threshold

    mock_high_confidence = MagicMock()
    mock_high_confidence.text = "valid clear speech"
    mock_high_confidence.no_speech_prob = 0.05
    mock_high_confidence.compression_ratio = 1.2
    mock_high_confidence.avg_logprob = -0.25  # Better than -1.0 threshold

    transcriber._model.transcribe = MagicMock(
        return_value=([mock_low_confidence, mock_high_confidence], None)
    )

    audio = np.ones(16000, dtype=np.float32) * 0.1
    result = transcriber.transcribe(audio)

    assert "mumbled" not in result
    assert "valid clear speech" in result


def test_whisper_bounded_audio_normalization():
    """Verify bounded audio normalization scales up audio with peak >= 0.03 and preserves floor < 0.03."""
    config = STTConfig()
    transcriber = WhisperTranscriber(config)

    captured_audios = []

    def mock_transcribe_call(audio_arr, **kwargs):
        captured_audios.append(np.copy(audio_arr))
        mock_seg = MagicMock()
        mock_seg.text = "test"
        mock_seg.no_speech_prob = 0.01
        mock_seg.compression_ratio = 1.0
        mock_seg.avg_logprob = -0.2
        return ([mock_seg], None)

    transcriber._model.transcribe = MagicMock(side_effect=mock_transcribe_call)

    # 1. Peak = 0.1 >= 0.03 -> gain = min(4.0, 0.90 / 0.1) = 4.0 -> peak becomes 0.4
    audio_moderate = np.ones(1000, dtype=np.float32) * 0.1
    transcriber.transcribe(audio_moderate)
    assert np.isclose(np.max(np.abs(captured_audios[0])), 0.4, atol=1e-5)
    assert transcriber.last_metrics["gain_applied"] == 4.0

    # 2. Peak = 0.5 >= 0.03 -> gain = min(4.0, 0.90 / 0.5) = 1.8 -> peak becomes 0.9
    audio_loud = np.ones(1000, dtype=np.float32) * 0.5
    transcriber.transcribe(audio_loud)
    assert np.isclose(np.max(np.abs(captured_audios[1])), 0.9, atol=1e-5)
    assert np.isclose(transcriber.last_metrics["gain_applied"], 1.8, atol=1e-5)

    # 3. Peak = 0.01 < 0.03 -> gain = 1.0 (no amplification of floor noise)
    audio_floor = np.ones(1000, dtype=np.float32) * 0.01
    transcriber.transcribe(audio_floor)
    assert np.isclose(np.max(np.abs(captured_audios[2])), 0.01, atol=1e-5)
    assert transcriber.last_metrics["gain_applied"] == 1.0


@pytest.mark.anyio
async def test_pipeline_rms_gating():
    """Verify audio chunks below min_rms_threshold are dropped before STT."""
    config = SonusConfig()
    config.vad.min_rms_threshold = 0.005
    pipeline = VoicePipeline(config)
    pipeline.stt.transcribe = MagicMock(return_value="Should not be called")

    # 1. Audio with RMS below threshold (0.001 < 0.005, 8000 samples = 0.5s at 16kHz)
    low_energy_audio = np.ones(8000, dtype=np.float32) * 0.001
    await pipeline._handle_user_utterance(low_energy_audio)
    assert pipeline.stt.transcribe.call_count == 0

    # 2. Audio with RMS above threshold (0.05 >= 0.005)
    normal_audio = np.ones(8000, dtype=np.float32) * 0.05

    async def mock_generate(**kwargs):
        if False:
            yield ""

    pipeline.llm.generate_sentences = mock_generate
    await pipeline._handle_user_utterance(normal_audio)
    assert pipeline.stt.transcribe.call_count == 1


@pytest.mark.anyio
async def test_pipeline_debug_audio_logging():
    """Verify debug_audio_dir creates WAV and JSON logs with Whisper metrics."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = SonusConfig()
        config.debug_audio_dir = tmp_dir
        pipeline = VoicePipeline(config)
        pipeline.stt.transcribe = MagicMock(return_value="what is json")
        pipeline.stt.last_metrics = {
            "peak_before_norm": 0.2,
            "gain_applied": 4.0,
            "segments": [{"text": "what is json", "no_speech_prob": 0.01, "avg_logprob": -0.2}],
        }

        async def mock_generate(**kwargs):
            if False:
                yield ""

        pipeline.llm.generate_sentences = mock_generate

        audio = np.ones(8000, dtype=np.float32) * 0.1
        await pipeline._handle_user_utterance(audio)

        # Check files in tmp_dir
        wav_files = list(Path(tmp_dir).glob("*.wav"))
        json_files = list(Path(tmp_dir).glob("*.json"))

        assert len(wav_files) == 1
        assert len(json_files) == 1

        import json
        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["transcription"] == "what is json"
        assert "metrics" in data
        assert data["metrics"]["peak_before_norm"] == 0.2

        # Dropped low-rms chunk is also logged
        low_audio = np.ones(8000, dtype=np.float32) * 0.001
        await pipeline._handle_user_utterance(low_audio)
        assert len(list(Path(tmp_dir).glob("*.json"))) == 2


def test_whisper_nan_metrics_filtering():
    """Verify segments with non-finite / NaN metrics are safely rejected as hallucinations."""
    config = STTConfig()
    transcriber = WhisperTranscriber(config)

    mock_nan_seg = MagicMock()
    mock_nan_seg.text = "corrupted hallucination segment"
    mock_nan_seg.no_speech_prob = float("nan")
    mock_nan_seg.compression_ratio = 1.0
    mock_nan_seg.avg_logprob = float("nan")

    mock_valid = MagicMock()
    mock_valid.text = "clean valid speech"
    mock_valid.no_speech_prob = 0.01
    mock_valid.compression_ratio = 1.2
    mock_valid.avg_logprob = -0.2

    transcriber._model.transcribe = MagicMock(
        return_value=([mock_nan_seg, mock_valid], None)
    )

    audio = np.ones(16000, dtype=np.float32) * 0.1
    result, metrics = transcriber.transcribe_with_info(audio)

    assert "corrupted hallucination" not in result
    assert "clean valid speech" in result
    assert metrics["segments"][0]["rejected"] is True
    assert metrics["segments"][1]["rejected"] is False


@pytest.mark.anyio
async def test_pipeline_status_restored_on_dropped_utterance():
    """Verify pipeline status returns to IDLE when utterances are dropped for low RMS or short burst."""
    statuses = []
    config = SonusConfig()
    pipeline = VoicePipeline(config, on_status=lambda s, d: statuses.append(s))

    # Simulate UI in LISTENING state
    pipeline.on_status("LISTENING", "User speaking...")

    # 1. Burst shorter than 400ms (100 samples)
    short_audio = np.ones(100, dtype=np.float32)
    await pipeline._handle_user_utterance(short_audio)
    assert statuses[-1] == "IDLE"

    # Simulate LISTENING again
    pipeline.on_status("LISTENING", "User speaking...")

    # 2. Audio below min_rms_threshold
    low_rms_audio = np.ones(8000, dtype=np.float32) * 0.001
    await pipeline._handle_user_utterance(low_rms_audio)
    assert statuses[-1] == "IDLE"


@pytest.mark.anyio
async def test_pipeline_nan_audio_dropped_safely():
    """Verify audio containing NaN/Inf is sanitized and dropped by RMS gate without crashing."""
    config = SonusConfig()
    pipeline = VoicePipeline(config)
    pipeline.stt.transcribe = MagicMock(return_value="Should not be called")

    corrupted_audio = np.array([np.nan, np.inf, -np.inf, 0.0] * 2000, dtype=np.float32)
    await pipeline._handle_user_utterance(corrupted_audio)

    assert pipeline.stt.transcribe.call_count == 0


def test_save_debug_record_unique_names():
    """Verify consecutive debug records generate unique non-colliding file names."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = SonusConfig()
        config.debug_audio_dir = tmp_dir
        pipeline = VoicePipeline(config)

        audio = np.ones(1000, dtype=np.float32) * 0.1
        pipeline._save_debug_record(audio, "first", {"idx": 1})
        pipeline._save_debug_record(audio, "second", {"idx": 2})

        wav_files = list(Path(tmp_dir).glob("*.wav"))
        json_files = list(Path(tmp_dir).glob("*.json"))

        assert len(wav_files) == 2
        assert len(json_files) == 2
        assert len({f.stem for f in wav_files}) == 2
