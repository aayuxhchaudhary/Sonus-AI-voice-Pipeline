"""
Silero VAD v5 ONNX execution engine with stateful recurrent memory.

Maintains recurrent hidden states and a trailing acoustic context buffer across
contiguous audio frames to satisfy Silero v5's temporal receptive field requirement.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import onnxruntime as ort

from sonus.config import VADConfig


class SileroVAD:
    """
    Stateful ONNX runtime wrapper for Silero Voice Activity Detection (v5).
    
    Architectural Invariant:
        Silero VAD v5 expects an input tensor of shape [batch, context + frame_samples].
        For 16kHz sampling, the model evaluates 512-sample (32ms) audio frames with a
        mandatory 64-sample (4ms) trailing historical context prepended, yielding an
        effective input shape of [1, 576]. Evaluating raw [1, 512] without context
        collapses the convolutional feature maps, causing near-zero speech probability.
    """

    def __init__(self, config: VADConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate

        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Silero VAD model graph missing at: {model_path}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        # Recurrent state tensor: shape [2, batch=1, hidden=128]
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr_tensor = np.array(sample_rate, dtype=np.int64)
        self._context_size = 64 if sample_rate == 16000 else 32
        self._context = np.zeros(self._context_size, dtype=np.float32)

    def reset_states(self) -> None:
        """
        Zero recurrent cell hidden states and historical context buffer.
        Must be invoked between completed conversational speech turns.
        """
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self._context_size, dtype=np.float32)

    def is_speech(self, audio_frame: np.ndarray) -> Tuple[bool, float]:
        """
        Evaluate acoustic speech probability for a discrete audio frame.
        
        Args:
            audio_frame: Monophonic float32 audio chunk (512 samples at 16kHz).
            
        Returns:
            Tuple[bool, float]: (is_speech_decision, posterior_confidence_score).
        """
        frame = np.asarray(audio_frame, dtype=np.float32).ravel()
        expected_frame_size = 512 if self.sample_rate == 16000 else 256

        if len(frame) == expected_frame_size + self._context_size:
            input_data = frame
            self._context = frame[-self._context_size:].copy()
        else:
            if len(frame) < expected_frame_size:
                frame = np.pad(frame, (0, expected_frame_size - len(frame)))
            elif len(frame) > expected_frame_size:
                frame = frame[:expected_frame_size]

            # Prepend historical 64-sample context: [64] + [512] -> [576]
            input_data = np.concatenate([self._context, frame], axis=0)
            self._context = frame[-self._context_size:].copy()

        audio_input = np.expand_dims(input_data, axis=0).astype(np.float32)

        ort_inputs = {
            "input": audio_input,
            "state": self._state,
            "sr": self._sr_tensor,
        }

        ort_outputs = self._session.run(None, ort_inputs)
        speech_prob = float(np.asarray(ort_outputs[0]).ravel()[0])
        self._state = ort_outputs[1]

        return (speech_prob >= self.config.threshold), speech_prob
