"""
Thread-safe circular audio buffer for pre-speech capture and streaming audio frames.
"""

from collections import deque
import threading
from typing import List
import numpy as np


class AudioRingBuffer:
    """Fixed-capacity ring buffer holding NumPy audio chunks."""

    def __init__(self, max_chunks: int):
        self._max_chunks = max_chunks
        self._buffer: deque[np.ndarray] = deque(maxlen=max_chunks)
        self._lock = threading.Lock()

    def append(self, chunk: np.ndarray) -> None:
        """Append an audio chunk (thread-safe)."""
        with self._lock:
            self._buffer.append(chunk.copy())

    def get_all(self) -> np.ndarray:
        """Return all buffered chunks concatenated as a single 1D float32 array."""
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            return np.concatenate(list(self._buffer))

    def clear(self) -> None:
        """Flush the buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
