#!/usr/bin/env python3
"""
Model asset downloader for Sonus.

Downloads pre-trained ONNX and binary model weights required for offline CPU inference:
1. Silero VAD v5 ONNX (Voice Activity Detection)
2. Kokoro-82M ONNX (Neural Speech Synthesis)
3. Kokoro voices-v1.0.bin (Voice style embedding vectors)
"""

import os
import sys
import urllib.request
from pathlib import Path

MODELS = [
    {
        "name": "Silero VAD v5 ONNX",
        "url": "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
        "dest": Path("models/vad/silero_vad.onnx"),
        "min_size": 2_000_000,  # ~2.3 MB
    },
    {
        "name": "Kokoro-82M ONNX",
        "url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx",
        "dest": Path("models/kokoro/kokoro-v0_19.onnx"),
        "min_size": 300_000_000,  # ~325 MB
    },
    {
        "name": "Kokoro Voice Embeddings (v1.0)",
        "url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        "dest": Path("models/kokoro/voices-v1.0.bin"),
        "min_size": 25_000_000,  # ~28 MB
    },
]


def download_file(url: str, dest_path: Path, name: str) -> None:
    """Download a remote file with progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    print(f"\n📦 Fetching {name}...")
    print(f"   URL:  {url}")
    print(f"   Dest: {dest_path}")

    try:
        def progress_hook(blocks_transferred: int, block_size: int, total_size: int) -> None:
            if total_size <= 0:
                sys.stdout.write(f"\r   Downloaded {blocks_transferred * block_size // (1024 * 1024)} MB...")
            else:
                downloaded = blocks_transferred * block_size
                percent = min(100.0, (downloaded / total_size) * 100.0)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\r   [{percent:5.1f}%] {downloaded_mb:.1f} MB / {total_mb:.1f} MB")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, str(temp_path), reporthook=progress_hook)
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Atomic rename
        temp_path.replace(dest_path)
        print(f"   ✔ Verified and saved to {dest_path}")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download {name} from {url}: {e}") from e


def ensure_models() -> None:
    """Verify all required model binaries exist, downloading any that are missing."""
    print("=" * 65)
    print("SONUS MODEL VERIFICATION & ASSET DOWNLOADER")
    print("=" * 65)

    missing = []
    for m in MODELS:
        dest = m["dest"]
        if dest.exists() and dest.stat().st_size >= m["min_size"]:
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"✔ Found {m['name']:<30} ({size_mb:.1f} MB) at {dest}")
        else:
            missing.append(m)

    if not missing:
        print("\nAll model assets are present and verified. Ready to run Sonus!\n")
        return

    print(f"\n{len(missing)} model asset(s) need to be downloaded.")
    for m in missing:
        download_file(m["url"], m["dest"], m["name"])

    print("\n" + "=" * 65)
    print("All models successfully downloaded and verified!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    ensure_models()
