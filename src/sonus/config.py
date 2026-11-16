"""
Configuration subsystem for the Sonus voice AI pipeline.

Defines validated configuration schemas for all pipeline sub-components:
hardware audio I/O, Silero VAD v5, faster-whisper CTranslate2 STT, Ollama LLM,
and Kokoro-82M ONNX TTS.
"""

from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    """
    Hardware audio stream parameters for input capture and output playback.
    
    Attributes:
        input_device: ALSA/PipeWire device index or name substring (None triggers auto-negotiation).
        output_device: Output device index or name (None routes to system default output).
        input_sample_rate: Acoustic input target rate in Hz (16 kHz standardized for VAD & STT).
        output_sample_rate: Acoustic output synthesis rate in Hz (24 kHz native for Kokoro TTS).
        channels: Monophonic audio channel count.
        frame_size_ms: Ingestion frame duration in milliseconds (32ms corresponds to 512 samples at 16kHz).
    """
    input_device: Optional[int | str] = Field(default=None, description="Input device index/name (None = auto-detect working mic)")
    output_device: Optional[int | str] = Field(default=None, description="Output device index/name (None = system default)")
    input_sample_rate: int = Field(default=16000, description="Target sampling rate for VAD & STT (Hz)")
    output_sample_rate: int = Field(default=24000, description="Sampling rate for Kokoro TTS output (Hz)")
    channels: int = Field(default=1, description="Mono audio channel count")
    frame_size_ms: int = Field(default=32, description="Duration of each audio frame in ms (512 samples at 16kHz)")


class VADConfig(BaseModel):
    """
    Acoustic Voice Activity Detection parameters governing turn-taking and barge-in preemption.
    
    Attributes:
        model_path: Filesystem path to the Silero VAD v5 ONNX model graph.
        threshold: Neural posterior speech probability cutoff [0.0, 1.0].
        min_silence_duration_ms: Trailing silence required to conclude speech turn and trigger STT.
        speech_pad_ms: Margin of audio retained prior to and following detected speech boundaries.
        min_rms_threshold: Energy floor below which frames are discarded to prevent ambient noise ingress.
    """
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
    min_rms_threshold: float = Field(
        default=0.005,
        description="Minimum RMS energy threshold below which frames/utterance are dropped as silence"
    )


class STTConfig(BaseModel):
    """
    Speech-to-Text inference parameters powered by faster-whisper on AVX-512 VNNI CPU.
    
    Defaults to base.en (~74M parameters) for superior phonetic and accent representation
    while maintaining sub-500ms CPU inference latency.
    """
    model_size: str = Field(default="base.en", description="Model variant (base.en, tiny.en, small.en, base)")
    device: str = Field(default="cpu", description="Inference device (cpu or cuda)")
    compute_type: str = Field(default="int8", description="Quantization type (int8 for CPU AVX-512 VNNI acceleration)")
    num_threads: int = Field(default=4, description="Inference CPU thread allocation")
    initial_prompt: Optional[str] = Field(
        default="Sonus AI conversational voice assistant. Topics: programming, Python, JSON, APIs, code, software engineering.",
        description="Acoustic prompt conditioning the autoregressive vocabulary distribution"
    )
    hotwords: Optional[str] = Field(
        default="JSON API Python code format",
        description="Lexical hotwords biasing beam search prefix scoring"
    )
    condition_on_previous_text: bool = Field(
        default=False,
        description="Condition decoding on previous transcription text (disabled to prevent hallucination attractor loops)"
    )
    repetition_penalty: float = Field(
        default=1.2,
        description="Multiplicative penalty suppressing repetitive token generation"
    )
    no_repeat_ngram_size: int = Field(
        default=3,
        description="Maximum duplicate n-gram window size"
    )
    compression_ratio_threshold: float = Field(
        default=2.4,
        description="Zlib compression ratio threshold for rejecting repetitive hallucinations"
    )
    log_prob_threshold: float = Field(
        default=-1.0,
        description="Average token log-probability floor below which segments are rejected"
    )
    no_speech_threshold: float = Field(
        default=0.6,
        description="Decoder silence probability threshold for segment rejection"
    )


class LLMConfig(BaseModel):
    """
    Local Large Language Model generation parameters via Ollama.
    """
    base_url: str = Field(default="http://localhost:11434", description="Ollama HTTP API base endpoint")
    model: str = Field(default="llama3.2:1b", description="Model identifier (e.g. llama3.2:1b, llama3.2:3b, qwen2.5:3b)")
    temperature: float = Field(default=0.3, description="Sampling temperature (0.3 provides grounded factual precision)")
    user_name: Optional[str] = Field(default=None, description="User persona identifier for conversational grounding (None = neutral assistant)")
    system_prompt: str = Field(
        default=(
            "You are Sonus, a helpful, natural, and concise voice AI assistant.\n"
            "Real-world ground truth: Today is {current_date}, the time is {current_time}, and the year is {current_year}.\n"
            "You are a software program running locally on the user's computer. You have no physical body, room, or surroundings; never describe fictional scenes or roleplay.\n"
            "If the user corrects you on any topic, accept the correction immediately and never argue.\n"
            "Be direct, conversational, friendly, and complete in 1 to 2 spoken sentences. Never use markdown, bullet points, or asterisks."
        ),
        description="Conversational system prompt template with runtime token expansion"
    )


class TTSConfig(BaseModel):
    """
    Neural Text-to-Speech synthesis parameters powered by Kokoro-82M ONNX.
    """
    model_path: Path = Field(
        default=Path("models/kokoro/kokoro-v0_19.onnx"),
        description="Filesystem path to Kokoro-82M ONNX computation graph"
    )
    voices_path: Path = Field(
        default=Path("models/kokoro/voices-v1.0.bin"),
        description="Filesystem path to binary style embedding vectors"
    )
    voice: str = Field(default="af_heart", description="Voice profile identifier (e.g. af_heart, af_bella, am_adam)")
    speed: float = Field(default=1.0, description="Speech cadence tempo multiplier")


class SonusConfig(BaseModel):
    """
    Root configuration container aggregating all pipeline subsystems.
    """
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    debug_audio_dir: Optional[str] = Field(
        default=None,
        description="Target directory for forensic audio dumps and RFC 8259 telemetry logs"
    )
