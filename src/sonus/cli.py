"""
Interactive terminal interface for the Sonus voice assistant with mic diagnostics.
"""

import argparse
import asyncio
from datetime import datetime
import signal
import sys
import time
import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from pathlib import Path
from sonus.config import SonusConfig
from sonus.pipeline import VoicePipeline

console = Console()


def render_banner(config: SonusConfig, active_device_name: str = "") -> None:
    """Display startup diagnostics and hardware configuration."""
    banner_text = Text()
    banner_text.append("🎙️  SONUS: Conversational Voice AI Pipeline\n", style="bold magenta")
    banner_text.append("━" * 55 + "\n", style="dim")
    banner_text.append("• Architecture:    ", style="bold cyan")
    banner_text.append("Cascaded Async Stream (VAD -> STT -> LLM -> TTS)\n")
    banner_text.append("• Hardware:        ", style="bold cyan")
    banner_text.append("Intel AVX-512 VNNI CPU (Zero GPU Required)\n")
    banner_text.append("• Input Mic:       ", style="bold cyan")
    banner_text.append(f"{active_device_name or 'Auto-Negotiated Hardware Mic'}\n", style="bold green")
    banner_text.append("• VAD Engine:      ", style="bold cyan")
    banner_text.append(f"Silero VAD v5 [threshold={config.vad.threshold}]\n")
    banner_text.append("• STT Model:       ", style="bold cyan")
    banner_text.append(f"faster-whisper [{config.stt.model_size} INT8]\n")
    banner_text.append("• LLM Engine:      ", style="bold cyan")
    banner_text.append(f"Ollama [{config.llm.model}]\n")
    banner_text.append("• TTS Synthesizer: ", style="bold cyan")
    banner_text.append(f"Kokoro-82M ONNX [{config.tts.voice} @ 24kHz]\n")
    banner_text.append("• User Profile:    ", style="bold cyan")
    banner_text.append(f"{config.llm.user_name}\n", style="bold green")
    banner_text.append("• Clock Grounding: ", style="bold cyan")
    banner_text.append(f"{datetime.now().strftime('%A, %B %d, %Y (%I:%M %p)')}\n", style="bold yellow")
    if config.debug_audio_dir:
        banner_text.append("• Debug Audio Dir: ", style="bold cyan")
        banner_text.append(f"{config.debug_audio_dir}\n", style="bold yellow")
    banner_text.append("━" * 55 + "\n", style="dim")
    banner_text.append("Speak into your mic. Barge-in is active. Press Ctrl+C to exit.", style="italic green")

    console.print(Panel(banner_text, border_style="magenta", expand=False))


_last_meter_time = 0.0


def clear_meter_line() -> None:
    """Clear transient live audio level meter from console line if on interactive terminal."""
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


def on_level_event(rms: float) -> None:
    """Live RMS audio level metering callback."""
    global _last_meter_time
    now = time.monotonic()
    if now - _last_meter_time < 0.08:  # ~12 FPS throttle to prevent console flooding
        return
    _last_meter_time = now

    if sys.stdout.isatty():
        meter_len = 16
        # Scale RMS: 0.001 (silence) to 0.2+ (normal/loud speech)
        normalized = min(1.0, max(0.0, rms * 5.0))
        filled = int(normalized * meter_len)
        bar = "█" * filled + "░" * (meter_len - filled)
        color = "\033[32m" if rms < 0.15 else "\033[33m" if rms < 0.25 else "\033[31m"
        reset = "\033[0m"
        sys.stdout.write(f"\r  Mic Level: {color}[{bar}]{reset} RMS: {rms:.4f} ")
        sys.stdout.flush()


def on_status_event(status: str, details: str) -> None:
    clear_meter_line()
    color_map = {
        "READY": "green",
        "LISTENING": "bold green",
        "TRANSCRIBING": "cyan",
        "THINKING": "bold blue",
        "SPEAKING": "bold yellow",
        "BARGE_IN": "bold red",
        "IDLE": "dim",
        "ERROR": "bold white on red",
        "STOPPED": "dim red",
    }
    style = color_map.get(status, "white")
    console.print(f"[{style}][{status:>12}][/{style}] {details}")


def on_transcription_event(text: str) -> None:
    clear_meter_line()
    console.print(f"\n[bold green]👤 You:[/bold green] {text}")


def on_response_event(text: str) -> None:
    clear_meter_line()
    console.print(f"[bold yellow]🤖 Sonus:[/bold yellow] {text}")


def run_mic_test(config: SonusConfig) -> None:
    """Diagnostic tool to record 3 seconds and verify hardware audio capture."""
    console.print("\n[bold cyan]=== Hardware Microphone Diagnostic ===[/bold cyan]")
    pipeline = VoicePipeline(config)
    dev, hw_sr, decimation = pipeline.audio._resolve_input_device_and_rate()
    try:
        dev_info = sd.query_devices(dev) if dev is not None else sd.query_devices(kind="input")
        dev_name = dev_info.get("name", "Default System Input") if isinstance(dev_info, dict) else "Default System Input"
    except Exception:
        dev_name = "Default System Input"

    console.print(f"Target Device:  [bold green][{dev if dev is not None else 'Default'}] {dev_name}[/bold green]")
    console.print(f"Sample Rate:    {hw_sr} Hz (Downsample ratio: {decimation}:1 -> 16000 Hz)")
    console.print("\n[bold yellow]Recording 3 seconds from your mic... Speak now![/bold yellow]")

    frames_to_record = int(hw_sr * 3.0)
    data = sd.rec(frames_to_record, samplerate=hw_sr, channels=1, device=dev, dtype="float32")
    sd.wait()

    data_16k = data[:, 0][::decimation] if decimation > 1 else data[:, 0]
    peak = float(np.max(np.abs(data_16k)))
    rms = float(np.sqrt(np.mean(data_16k**2)))

    # Compute VAD score on loudest frame
    vad_score = 0.0
    for i in range(0, len(data_16k) - 512 + 1, 512):
        _, prob = pipeline.vad.is_speech(data_16k[i:i+512])
        if prob > vad_score:
            vad_score = prob

    console.print(f"• Peak Amplitude: [bold {'green' if peak > 0.05 else 'red'}]{peak:.4f}[/]")
    console.print(f"• RMS Energy:     [bold {'green' if rms > 0.005 else 'red'}]{rms:.4f}[/]")
    console.print(f"• Peak VAD Score: [bold {'green' if vad_score > 0.3 else 'yellow'}]{vad_score:.4f}[/]")

    if peak < 0.01:
        console.print("[bold red]❌ No audio detected. Microphone is likely muted or disconnected in ALSA/PipeWire.[/bold red]")
        return

    console.print("\n[cyan]Transcribing test audio with Whisper...[/cyan]")
    text, metrics = pipeline.stt.transcribe_with_info(data_16k)
    if config.debug_audio_dir:
        pipeline._save_debug_record(
            audio_data=data_16k,
            text=text,
            metrics=metrics,
        )
        console.print(f"[cyan]Forensic recording saved to {config.debug_audio_dir}[/cyan]")
    if text:
        console.print(f"[bold green]Whisper Heard:[/bold green] \"{text}\"")
        console.print("[green]✔ Audio pipeline verified successfully![/green]\n")
    else:
        console.print("[yellow]Whisper detected sound, but no intelligible words were transcribed.[/yellow]\n")


async def async_main(config: SonusConfig) -> None:
    pipeline = VoicePipeline(
        config=config,
        on_status=on_status_event,
        on_transcription=on_transcription_event,
        on_response=on_response_event,
        on_level=on_level_event,
    )

    try:
        dev, _, _ = pipeline.audio._resolve_input_device_and_rate()
        dev_info = sd.query_devices(dev) if dev is not None else sd.query_devices(kind="input")
        dev_name = dev_info.get("name", "") if isinstance(dev_info, dict) else ""
    except Exception:
        dev_name = "Auto-Negotiated Hardware Mic"
    render_banner(config, active_device_name=dev_name)

    loop = asyncio.get_running_loop()

    def handle_signal():
        console.print("\n[yellow]Shutdown requested. Exiting gracefully...[/yellow]")
        pipeline.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            pass

    await pipeline.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sonus: Production-grade, low-latency conversational voice AI pipeline running on CPU."
    )
    parser.add_argument("--test-mic", action="store_true", help="Run 3-second microphone capture and transcription diagnostic")
    parser.add_argument(
        "--user-name",
        type=str,
        default=None,
        help="User persona name for conversational context and identity grounding (default: Aayush)",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default=None,
        help="Whisper model variant (base.en, tiny.en, small.en, base) [default: base.en]",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Ollama LLM model identifier (llama3.2:1b, llama3.2:3b, qwen2.5:3b) [default: llama3.2:1b]",
    )
    parser.add_argument(
        "--tts-voice",
        type=str,
        default=None,
        help="Kokoro voice embedding profile (af_heart, af_bella, am_adam) [default: af_heart]",
    )
    parser.add_argument(
        "--debug-audio-dir",
        type=str,
        default=None,
        help="Filesystem directory to write forensic audio WAV captures and RFC 8259 telemetry logs",
    )
    args = parser.parse_args()

    config = SonusConfig()
    if args.stt_model:
        config.stt.model_size = args.stt_model

    if args.llm_model:
        config.llm.model = args.llm_model

    if args.tts_voice:
        config.tts.voice = args.tts_voice

    if args.user_name:
        config.llm.user_name = args.user_name
        config.stt.initial_prompt = (
            f"Sonus AI conversational voice assistant with {args.user_name}. "
            f"Topics: {args.user_name}, programming, Python, JSON, APIs, code, software engineering."
        )
        config.stt.hotwords = f"{args.user_name} JSON API Python code format"

    if args.debug_audio_dir:
        config.debug_audio_dir = str(Path(args.debug_audio_dir).expanduser())

    if args.test_mic:
        run_mic_test(config)
        return

    try:
        asyncio.run(async_main(config))
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)


if __name__ == "__main__":
    main()
