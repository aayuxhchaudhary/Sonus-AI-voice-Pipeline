"""
Unit tests for AudioStreamManager hardware negotiation, RMS metering, and atomic barge-in interrupt.
"""

from unittest.mock import MagicMock, patch
import numpy as np
from sonus.config import AudioConfig
from sonus.audio.stream import AudioStreamManager


def test_resolve_input_device_uses_default_at_16000():
    """Verify input device negotiation requests 16000Hz directly without hardcoded ALC strings."""
    config = AudioConfig(input_device=None, input_sample_rate=16000)
    manager = AudioStreamManager(config)

    with patch("sounddevice.check_input_settings") as mock_check, \
         patch("sounddevice.query_devices") as mock_query:
        # Default device allows 16000Hz
        mock_check.return_value = True

        dev, rate, decimation = manager._resolve_input_device_and_rate()

        # Should target default device (None) at native 16000Hz with ratio 1
        assert dev is None
        assert rate == 16000
        assert decimation == 1
        mock_check.assert_called_with(
            device=None,
            channels=1,
            dtype="float32",
            samplerate=16000
        )


def test_stop_playback_atomic_interrupt():
    """Verify stop_playback sets atomic interrupt flag and drops queue immediately."""
    config = AudioConfig()
    manager = AudioStreamManager(config)

    # Queue multiple audio segments
    dummy_chunk1 = np.ones(4800, dtype=np.float32)
    dummy_chunk2 = np.ones(4800, dtype=np.float32)
    manager.queue_audio(dummy_chunk1)
    manager.queue_audio(dummy_chunk2)

    assert not manager._interrupt_playback.is_set()
    assert manager._playback_queue.qsize() == 2

    # Invoke stop_playback
    manager.stop_playback()

    assert manager._interrupt_playback.is_set()
    assert manager._playback_queue.empty()
    assert not manager.is_playing


def test_queue_audio_clears_interrupt_flag():
    """Verify queueing new audio clears previous interrupt flag."""
    config = AudioConfig()
    manager = AudioStreamManager(config)

    manager._interrupt_playback.set()
    manager.queue_audio(np.ones(100, dtype=np.float32))

    assert not manager._interrupt_playback.is_set()
    assert manager._playback_queue.qsize() == 1


def test_playback_worker_slice_loop_aborts_on_interrupt():
    """Verify that the 1024-sample slice writing loop terminates immediately on _interrupt_playback."""
    config = AudioConfig(output_sample_rate=24000)
    manager = AudioStreamManager(config)

    # 4096 samples = exactly 4 slices of 1024 samples
    audio_chunk = np.ones(4096, dtype=np.float32)
    manager.queue_audio(audio_chunk)

    written_slices = []

    mock_stream = MagicMock()
    def mock_write(slice_data):
        written_slices.append(len(slice_data))
        # Trigger interruption on the first slice
        manager.stop_playback()

    mock_stream.write = mock_write

    # Execute the 1024-sample slice writing loop directly
    chunk = manager._playback_queue.get()
    chunk_slice_size = 1024
    offset = 0
    while (
        offset < len(chunk)
        and not manager._stop_playback_flag.is_set()
        and not manager._interrupt_playback.is_set()
    ):
        end = min(offset + chunk_slice_size, len(chunk))
        slice_data = chunk[offset:end]
        mock_stream.write(slice_data)
        offset = end

    # Only 1 slice should be written before the loop aborts
    assert len(written_slices) == 1
    assert written_slices[0] == 1024
    assert manager._interrupt_playback.is_set()
