"""
Configuration management for Sonus voice pipeline.
"""

from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    """Audio sampling and hardware configuration."""
    input_device: Optional[int | str] = Field(default=None, description="Input device index/name (None = auto-detect working mic)")
    output_device: Optional[int | str] = Field(default=None, description="Output device index/name (None = system default)")
    input_sample_rate: int = Field(default=16000, description="Target sampling rate for VAD & STT (Hz)")
    output_sample_rate: int = Field(default=24000, description="Sampling rate for Kokoro TTS output (Hz)")
    channels: int = Field(default=1, description="Mono audio channel count")
    frame_size_ms: int = Field(default=32, description="Duration of each audio frame in ms (512 samples at 16kHz)")


class VADConfig(BaseModel):
    """Voice Activity Detection configuration using Silero VAD v5."""
    model_path: Path = Field(
        default=Path("models/vad/silero_vad.onnx"),
        description="Path to Silero VAD ONNX model"
    )
    threshold: float = Field(default=0.35, description="Speech probability threshold (0.0 to 1.0)")
    min_silence_duration_ms: int = Field(
        default=450,
        description="Silence duration required to conclude the user finished speaking"
    )
    speech_pad_ms: int = Field(
        default=60,
        description="Audio padding added before and after speech in ms"
    )


class STTConfig(BaseModel):
    """Speech-to-Text configuration using faster-whisper."""
    model_size: str = Field(default="tiny.en", description="Model variant (tiny.en, base.en, small.en)")
    device: str = Field(default="cpu", description="Inference device (cpu or cuda)")
    compute_type: str = Field(default="int8", description="Quantization type (int8 for CPU AVX-512)")
    num_threads: int = Field(default=4, description="Inference CPU threads")


class LLMConfig(BaseModel):
    """LLM engine configuration (Ollama / Local HTTP endpoint)."""
    base_url: str = Field(default="http://localhost:11434", description="Ollama API endpoint")
    model: str = Field(default="llama3.2:1b", description="Model identifier")
    temperature: float = Field(default=0.6, description="Sampling temperature")
    system_prompt: str = Field(
        default=(
            "You are Sonus, a sharp, concise, and natural voice assistant. "
            "You are in a live voice conversation with the user. "
            "Keep your responses direct, natural, conversational, and typically under 2 to 3 sentences. "
            "Never use markdown formatting, bullet points, asterisks, or emoji, as your output is spoken aloud."
        ),
        description="Conversational system prompt for voice synthesis"
    )


class TTSConfig(BaseModel):
    """Text-to-Speech configuration using Kokoro-82M ONNX."""
    model_path: Path = Field(
        default=Path("models/kokoro/kokoro-v0_19.onnx"),
        description="Path to Kokoro ONNX model"
    )
    voices_path: Path = Field(
        default=Path("models/kokoro/voices-v1.0.bin"),
        description="Path to binary voice embeddings file"
    )
    voice: str = Field(default="af_heart", description="Voice profile identifier (e.g. af_heart, af_bella, am_adam)")
    speed: float = Field(default=1.0, description="Speech cadence speed multiplier")


class SonusConfig(BaseModel):
    """Root configuration aggregator."""
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
