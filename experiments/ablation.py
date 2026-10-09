"""Chapter 3.5/3.7: six-module ablations and sensitivity sweeps."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import run_extended_ablations

if __name__ == "__main__":
    raise SystemExit(run_extended_ablations.main())
