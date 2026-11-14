# 🎙️ Sonus

> **Production-Grade, Ultra-Low-Latency Conversational Voice AI Pipeline Optimized for CPU.**  
> *Zero GPUs required. Built for real-time speech interaction with native barge-in interruption.*

---

## 🌟 Overview

**Sonus** is an end-to-end voice-to-voice conversational agent that runs completely locally on consumer x86_64 CPUs (leveraging Intel **AVX-512** and **VNNI** instructions). It replaces heavy, resource-intensive GPU audio foundation models with a decoupled, asynchronous streaming pipeline capable of sub-second turnaround times.

```
                  ┌─────────────────────────────────────────────────────────────┐
                  │                 SONUS CASCADED PIPELINE                     │
                  └─────────────────────────────────────────────────────────────┘

  [ Microphone ] ──► (1. Silero VAD v5)  ──► [ Speech Chunks ]
                            │ (32ms frames @ 16kHz)
                            ▼
                     (2. faster-whisper) ──► [ User Transcription ]
                            │ (CTranslate2 INT8)
                            ▼
                     (3. Ollama LLM)     ──► [ Token Stream ]
                            │ (Llama-3.2-1B)
                            ▼
                     [ Sentence Chunker ] ──► [ First Sentence Ready ]
                            │
                            ▼
                     (4. Kokoro-82M ONNX)──► (5. Audio Stream Manager) ──► [ Speaker ]
                       (Studio 24kHz Audio)    (Interruptible Playback)
```

---

## ⚡ Performance & Benchmarks (Intel i7-1185G7 AVX-512)

| Pipeline Stage | Model / Engine | Footprint | Real-Time Latency |
| :--- | :--- | :--- | :--- |
| **VAD (Voice Activity)** | Silero VAD v5 (ONNX) | ~2.3 MB | **< 2ms** per 32ms frame |
| **STT (Speech-to-Text)** | faster-whisper (`tiny.en` INT8) | ~75 MB | **~120ms – 180ms** per utterance |
| **LLM (Reasoning Brain)** | Llama-3.2-1B-Instruct (Q4_K_M) | ~1.3 GB | **35–55 tokens/sec** (First sentence: ~150ms) |
| **TTS (Speech Synthesis)**| Kokoro-82M (ONNX) | ~310 MB | **~150ms – 250ms** per sentence (24kHz) |
| **Barge-in (Interruption)**| State Machine Buffer Flush | 0 MB | **< 10ms** instant audio mute |
| **Total Turnaround Time** | *User stops speaking -> AI audio starts* | **~4 GB RAM** | **~750ms – 1.1s** |

---

## 📁 Repository Structure

```
sonus/
├── models/
│   ├── vad/silero_vad.onnx        # Silero VAD v5 ONNX model
│   └── kokoro/
│       ├── kokoro-v0_19.onnx      # Kokoro-82M ONNX synthesis graph
│       └── voices-v1.0.bin        # 54 studio voice embeddings
├── src/
│   └── sonus/
│       ├── audio/
│       │   ├── ring_buffer.py     # Thread-safe circular pre-speech audio buffer
│       │   └── stream.py          # sounddevice non-blocking capture & interruptible player
│       ├── vad/
│       │   └── silero.py          # Pure ONNX Silero VAD v5 stateful wrapper
│       ├── stt/
│       │   └── transcriber.py     # faster-whisper CTranslate2 INT8 CPU transcriber
│       ├── llm/
│       │   └── client.py          # Async streaming Ollama client with sentence boundary chunker
│       ├── tts/
│       │   └── synthesizer.py     # Kokoro-82M ONNX 24kHz audio synthesis engine
│       ├── config.py              # Strongly-typed Pydantic pipeline settings
│       ├── pipeline.py            # Event-driven async orchestrator tying all stages
│       └── cli.py                 # Rich terminal interface with state badges
├── tests/                         # Pytest test suite (VAD, STT, LLM, TTS)
├── main.py                        # Application entrypoint
└── pyproject.toml                 # Standard packaging and dependencies
```

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (Managed automatically via `uv`).
- PortAudio / PipeWire audio subsystem.
- [Ollama](https://ollama.com) running locally with `llama3.2:1b`.

### 2. Start the LLM Engine
Ensure the Ollama daemon is active and has pulled the model:
```bash
ollama serve
ollama pull llama3.2:1b
```

### 3. Run the Voice Assistant
Activate the virtual environment and execute `main.py`:
```bash
# Using the pre-configured Python 3.12 venv
./.venv/bin/python main.py
```

---

## 🎙️ Key Features

### 1. Native Barge-in Interruption
You never have to wait for the assistant to finish talking. When the user starts speaking while the AI is outputting audio, Silero VAD detects speech instantly, mutes active speaker playback, cancels any pending downstream LLM/TTS generation tasks, and transitions immediately back to listening mode.

### 2. Sentence-Pipelined Streaming
Instead of waiting for the full LLM answer before generating speech, Sonus parses sentences on the fly (`.`, `?`, `!`) and feeds them into Kokoro-82M immediately. The first sentence begins playing in your speakers while subsequent sentences are being generated in the background.

### 3. Studio Audio Quality on CPU
Kokoro-82M outputs rich, 24,000Hz mono audio without robotic artifacts, outperforming older engines like eSpeak or Festival while using negligible CPU memory.

---

## 🧪 Testing & Verification

Run the test suite across all four model components:
```bash
PYTHONPATH=src ./.venv/bin/pytest -v tests/
```

Expected output:
```
tests/test_llm.py::test_ollama_streaming[asyncio] PASSED
tests/test_stt.py::test_whisper_transcriber_silence PASSED
tests/test_tts.py::test_kokoro_tts_synthesis PASSED
tests/test_vad.py::test_silero_vad_synthetic_silence PASSED
tests/test_vad.py::test_silero_vad_reset_state PASSED
5 passed in ~3s
```

---

## 📜 License
MIT License. Open-source models used:
- Silero VAD (MIT)
- faster-whisper / OpenAI Whisper (MIT)
- LLaMA 3.2 (Meta LLaMA 3.2 Community License)
- Kokoro-82M (Apache 2.0)
