"""Chapter 3.3/3.5: full-data 3-backbone x 4-condition experiments."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import run_main

if __name__ == "__main__":
    raise SystemExit(run_main.main())
