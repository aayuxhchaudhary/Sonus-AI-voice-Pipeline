"""
Silero VAD v5 ONNX implementation with state management and silence thresholding.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import onnxruntime as ort

from sonus.config import VADConfig


class SileroVAD:
    """Stateful ONNX runtime wrapper for Silero VAD v5."""

    def __init__(self, config: VADConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate

        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Silero VAD model not found at {model_path}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr_tensor = np.array(sample_rate, dtype=np.int64)

    def reset_states(self) -> None:
        """Reset internal recurrent state."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, audio_frame: np.ndarray) -> Tuple[bool, float]:
        """
        Evaluate single audio chunk.
        
        Args:
            audio_frame: 1D NumPy array of float32 samples (512 samples for 16kHz).
            
        Returns:
            Tuple[bool, float]: (is_speech_flag, confidence_score)
        """
        if audio_frame.ndim == 1:
            audio_input = np.expand_dims(audio_frame, axis=0).astype(np.float32)
        else:
            audio_input = audio_frame.astype(np.float32)

        ort_inputs = {
            "input": audio_input,
            "state": self._state,
            "sr": self._sr_tensor,
        }

        ort_outputs = self._session.run(None, ort_inputs)
        speech_prob = float(ort_outputs[0][0][0])
        self._state = ort_outputs[1]

        return (speech_prob >= self.config.threshold), speech_prob
