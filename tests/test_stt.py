"""
Unit tests for faster-whisper INT8 CPU transcription.
"""

import numpy as np
from sonus.config import STTConfig
from sonus.stt.transcriber import WhisperTranscriber


def test_whisper_transcriber_silence():
    config = STTConfig(model_size="tiny.en")
    transcriber = WhisperTranscriber(config)

    # 1 second of silence at 16kHz
    silence = np.zeros(16000, dtype=np.float32)
    result = transcriber.transcribe(silence)
    assert isinstance(result, str)
