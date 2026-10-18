"""Makes ``src`` importable when a tool is run directly from a checkout (``python tools/<tool>.py``); not needed after ``pip install -e .``."""
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
