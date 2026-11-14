"""
Sonus: Production-grade, low-latency conversational voice agent for CPU.
"""

__version__ = "0.1.0"
__author__ = "Aayush"

from sonus.config import SonusConfig
from sonus.pipeline import VoicePipeline

__all__ = ["SonusConfig", "VoicePipeline", "__version__"]
