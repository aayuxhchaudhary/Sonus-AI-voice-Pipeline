"""
Unit tests for VoicePipeline barge-in cancellation, mutex locking, echo suppression, and silence trimming.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

try:
    import pytest
except ImportError:
    class _MockPytest:
        class mark:
            anyio = lambda fn: fn
    pytest = _MockPytest()  # type: ignore

from sonus.config import SonusConfig
from sonus.pipeline import VoicePipeline


@pytest.mark.anyio
async def test_pipeline_barge_in_cancellation():
    """Verify barge-in properly cancels the active asyncio generation task on the event loop."""
    config = SonusConfig()
    pipeline = VoicePipeline(config)
    pipeline._running = True
    pipeline._event_loop = asyncio.get_running_loop()

    # Simulate an active long-running LLM/TTS generation task
    task_cancelled = False

    async def long_generation():
        nonlocal task_cancelled
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            task_cancelled = True
            raise

    pipeline._processing_task = pipeline._event_loop.create_task(long_generation())

    # Simulate speaker currently playing audio
    pipeline.audio._is_playing.set()

    # Audio callback with speech above barge-in threshold (e.g. 0.85)
    pipeline.vad.is_speech = MagicMock(return_value=(True, 0.85))

    speech_chunk = np.ones(512, dtype=np.float32)
    pipeline._audio_callback(speech_chunk)

    # Allow event loop to process scheduled cancellation
    await asyncio.sleep(0.05)

    assert task_cancelled
    assert pipeline._processing_task.cancelled()
    assert not pipeline.audio.is_playing
    assert pipeline._is_speaking_user
    assert len(pipeline._speech_chunks) == 1
    assert np.allclose(pipeline.vad._context, speech_chunk[-64:])


def test_pipeline_echo_suppression_during_playback():
    """Verify moderate audio (speaker bleed) during playback is suppressed and pre-speech buffer is cleared."""
    config = SonusConfig()
    pipeline = VoicePipeline(config)
    pipeline._running = True
    pipeline._event_loop = MagicMock()
    pipeline._event_loop.is_running.return_value = True

    # Speaker is active
    pipeline.audio._is_playing.set()
    # Add dummy item to pre-speech buffer
    pipeline._pre_speech_buffer.append(np.ones(512, dtype=np.float32))

    # Moderate VAD score from speaker bleed (0.45 < barge_in_threshold of 0.65)
    pipeline.vad.is_speech = MagicMock(return_value=(True, 0.45))

    chunk = np.ones(512, dtype=np.float32) * 0.1
    pipeline._audio_callback(chunk)

    # Pre-speech buffer must be cleared to suppress echo
    assert pipeline._pre_speech_buffer.get_all().size == 0
    # User should NOT be marked as speaking
    assert not pipeline._is_speaking_user
    # Playback should NOT be interrupted
    assert pipeline.audio.is_playing


def test_pipeline_trailing_silence_trimming():
    """Verify trailing silence chunks beyond speech_pad_ms are trimmed before scheduling STT."""
    config = SonusConfig()
    config.audio.frame_size_ms = 32
    config.vad.min_silence_duration_ms = 320   # 10 chunks of silence
    config.vad.speech_pad_ms = 64              # Keep only 2 chunks of pad

    pipeline = VoicePipeline(config)
    pipeline._running = True
    pipeline._event_loop = MagicMock()
    pipeline._event_loop.is_running.return_value = True

    scheduled_audios = []
    pipeline._schedule_utterance_processing = MagicMock(side_effect=lambda a: scheduled_audios.append(a))

    current_time = 1000.0

    def mock_time():
        return current_time

    with patch("time.monotonic", side_effect=mock_time):
        # 1. Start speaking (speech chunk)
        pipeline.vad.is_speech = MagicMock(return_value=(True, 0.9))
        pipeline._audio_callback(np.ones(512, dtype=np.float32) * 0.8)
        assert pipeline._is_speaking_user

        # 2. Provide 12 silence chunks (384ms), advancing time by 32ms per frame
        pipeline.vad.is_speech = MagicMock(return_value=(False, 0.05))
        for _ in range(12):
            current_time += 0.032
            pipeline._audio_callback(np.zeros(512, dtype=np.float32))

    # Utterance should be concluded and trimmed
    assert not pipeline._is_speaking_user
    assert len(scheduled_audios) == 1

    audio_data = scheduled_audios[0]
    # Total chunks fed: 1 speech chunk + 12 silence chunks = 13 chunks (6656 samples)
    # With trimming, excess silence chunks beyond the 2 pad chunks (64ms) are discarded
    # Remaining chunks: 1 speech + 2 pad = 3 or 4 chunks (<= 2048 samples)
    assert len(audio_data) <= 512 * 4


@pytest.mark.anyio
async def test_pipeline_utterance_lock_prevents_race_conditions():
    """Verify _utterance_lock serializes utterance handling and protects conversation history."""
    config = SonusConfig()
    pipeline = VoicePipeline(config)
    pipeline._running = True

    execution_order = []

    def mock_transcribe(audio):
        return "Test utterance"

    pipeline.stt.transcribe = mock_transcribe

    async def mock_generate(user_message, conversation_history):
        execution_order.append("start")
        await asyncio.sleep(0.05)
        execution_order.append("end")
        yield "Response"

    pipeline.llm.generate_sentences = mock_generate
    pipeline.tts.synthesize = MagicMock(return_value=(np.zeros(100, dtype=np.float32), 24000))

    audio1 = np.ones(8000, dtype=np.float32)
    audio2 = np.ones(8000, dtype=np.float32)

    # Launch two utterances concurrently
    await asyncio.gather(
        pipeline._handle_user_utterance(audio1),
        pipeline._handle_user_utterance(audio2),
    )

    # Because of the lock, generation must run strictly sequential: start -> end -> start -> end
    assert execution_order == ["start", "end", "start", "end"]
    # History should have 4 records: user, assistant, user, assistant
    assert len(pipeline._conversation_history) == 4
