"""
Standalone test runner for Sonus voice pipeline unit and integration tests.
Runs all tests with evidence output.
"""

import os
import sys

# If run under an interpreter where binary wheels mismatch (e.g. host python 3.15 vs venv python 3.12),
# re-exec into the project's virtualenv python to execute the real test suite with native extensions.
venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "bin", "python3"))
if os.path.exists(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
    try:
        import onnxruntime
        import ctranslate2
        if not hasattr(ctranslate2.models, "Whisper"):
            raise ImportError("ctranslate2 missing Whisper extension")
    except Exception:
        os.execv(venv_python, [venv_python] + sys.argv)

import asyncio
from unittest.mock import MagicMock

sys.path.insert(0, ".")
sys.path.insert(0, "src")
sys.path.append(".venv/lib/python3.12/site-packages")

# Gracefully handle binary dependencies: attempt real import first, fall back to mock if ABI mismatch
for mod_name in ["sounddevice", "onnxruntime", "faster_whisper", "kokoro_onnx"]:
    try:
        __import__(mod_name)
    except Exception:
        sys.modules[mod_name] = MagicMock()

from tests.test_llm import (
    test_sentence_splitting_abbreviations,
    test_sentence_splitting_decimals_and_currency,
    test_sentence_splitting_initials_and_acronyms,
    test_sentence_splitting_acronyms_before_capitals,
    test_sentence_splitting_numbered_lists,
    test_sentence_splitting_quotes_and_brackets,
    test_sentence_splitting_incremental_streaming,
    test_system_prompt_grounding,
    test_sanitize_spoken_text,
    test_ollama_streaming,
)
from tests.test_stream import (
    test_resolve_input_device_uses_default_at_16000,
    test_stop_playback_atomic_interrupt,
    test_queue_audio_clears_interrupt_flag,
    test_playback_worker_slice_loop_aborts_on_interrupt,
)
from tests.test_pipeline import (
    test_pipeline_barge_in_cancellation,
    test_pipeline_echo_suppression_during_playback,
    test_pipeline_trailing_silence_trimming,
    test_pipeline_utterance_lock_prevents_race_conditions,
)
from tests.test_stt import (
    test_whisper_transcriber_silence,
    test_whisper_no_speech_prob_filtering,
    test_whisper_transcriber_real_speech,
    test_whisper_initial_prompt_and_hotwords,
    test_whisper_compression_ratio_filtering,
    test_whisper_log_prob_filtering,
    test_whisper_bounded_audio_normalization,
    test_pipeline_rms_gating,
    test_pipeline_debug_audio_logging,
    test_whisper_nan_metrics_filtering,
    test_pipeline_status_restored_on_dropped_utterance,
    test_pipeline_nan_audio_dropped_safely,
    test_save_debug_record_unique_names,
)
from tests.test_vad import (
    test_silero_vad_synthetic_silence,
    test_silero_vad_reset_state,
    test_silero_vad_context_prepended,
    test_silero_vad_real_speech_samples,
)
from tests.test_tts import test_kokoro_tts_synthesis

test_functions = [
    ("test_sentence_splitting_abbreviations", test_sentence_splitting_abbreviations, False),
    ("test_sentence_splitting_decimals_and_currency", test_sentence_splitting_decimals_and_currency, False),
    ("test_sentence_splitting_initials_and_acronyms", test_sentence_splitting_initials_and_acronyms, False),
    ("test_sentence_splitting_acronyms_before_capitals", test_sentence_splitting_acronyms_before_capitals, False),
    ("test_sentence_splitting_numbered_lists", test_sentence_splitting_numbered_lists, False),
    ("test_sentence_splitting_quotes_and_brackets", test_sentence_splitting_quotes_and_brackets, False),
    ("test_sentence_splitting_incremental_streaming", test_sentence_splitting_incremental_streaming, False),
    ("test_system_prompt_grounding", test_system_prompt_grounding, False),
    ("test_sanitize_spoken_text", test_sanitize_spoken_text, False),
    ("test_ollama_streaming", test_ollama_streaming, True),
    ("test_resolve_input_device_uses_default_at_16000", test_resolve_input_device_uses_default_at_16000, False),
    ("test_stop_playback_atomic_interrupt", test_stop_playback_atomic_interrupt, False),
    ("test_queue_audio_clears_interrupt_flag", test_queue_audio_clears_interrupt_flag, False),
    ("test_playback_worker_slice_loop_aborts_on_interrupt", test_playback_worker_slice_loop_aborts_on_interrupt, False),
    ("test_pipeline_barge_in_cancellation", test_pipeline_barge_in_cancellation, True),
    ("test_pipeline_echo_suppression_during_playback", test_pipeline_echo_suppression_during_playback, False),
    ("test_pipeline_trailing_silence_trimming", test_pipeline_trailing_silence_trimming, False),
    ("test_pipeline_utterance_lock_prevents_race_conditions", test_pipeline_utterance_lock_prevents_race_conditions, True),
    ("test_whisper_transcriber_silence", test_whisper_transcriber_silence, False),
    ("test_whisper_no_speech_prob_filtering", test_whisper_no_speech_prob_filtering, False),
    ("test_whisper_transcriber_real_speech", test_whisper_transcriber_real_speech, False),
    ("test_whisper_initial_prompt_and_hotwords", test_whisper_initial_prompt_and_hotwords, False),
    ("test_whisper_compression_ratio_filtering", test_whisper_compression_ratio_filtering, False),
    ("test_whisper_log_prob_filtering", test_whisper_log_prob_filtering, False),
    ("test_whisper_bounded_audio_normalization", test_whisper_bounded_audio_normalization, False),
    ("test_pipeline_rms_gating", test_pipeline_rms_gating, True),
    ("test_pipeline_debug_audio_logging", test_pipeline_debug_audio_logging, True),
    ("test_whisper_nan_metrics_filtering", test_whisper_nan_metrics_filtering, False),
    ("test_pipeline_status_restored_on_dropped_utterance", test_pipeline_status_restored_on_dropped_utterance, True),
    ("test_pipeline_nan_audio_dropped_safely", test_pipeline_nan_audio_dropped_safely, True),
    ("test_save_debug_record_unique_names", test_save_debug_record_unique_names, False),
    ("test_silero_vad_synthetic_silence", test_silero_vad_synthetic_silence, False),
    ("test_silero_vad_reset_state", test_silero_vad_reset_state, False),
    ("test_silero_vad_context_prepended", test_silero_vad_context_prepended, False),
    ("test_silero_vad_real_speech_samples", test_silero_vad_real_speech_samples, False),
    ("test_kokoro_tts_synthesis", test_kokoro_tts_synthesis, False),
]

passed = 0
failed = 0

print("=" * 65)
print("RUNNING SONUS VERIFICATION TEST SUITE")
print("=" * 65)

for name, func, is_async in test_functions:
    try:
        if is_async:
            asyncio.run(func())
        else:
            func()
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name}: {e}")
        failed += 1

print("=" * 65)
print(f"RESULTS: {passed} PASSED, {failed} FAILED (TOTAL {len(test_functions)})")
print("=" * 65)

if failed > 0:
    sys.exit(1)
