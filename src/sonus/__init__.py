"""
Sonus: Production-grade, low-latency conversational voice agent for CPU.
"""

__version__ = "0.1.0"
__author__ = "Aayush"

from sonus.config import SonusConfig

try:
    from sonus.pipeline import VoicePipeline
except (ImportError, OSError):
    VoicePipeline = None  # type: ignore

__all__ = ["SonusConfig", "VoicePipeline", "__version__"]

