"""Compatibility wrapper for legacy imports."""
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gemma_trader.dashboard import *  # noqa: F401,F403
if __name__ == "__main__":
    from gemma_trader.dashboard import main
    main()
