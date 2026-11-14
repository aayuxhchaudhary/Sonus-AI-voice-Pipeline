#!/usr/bin/env python3
"""
Sonus Entrypoint
"""

import sys
from pathlib import Path

# Ensure src/ is on Python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from sonus.cli import main

if __name__ == "__main__":
    main()
