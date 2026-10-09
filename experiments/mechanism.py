"""Additional mechanism cases: queries, candidates, weights, gates and scores."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import generate_mechanism_cases

if __name__ == "__main__":
    raise SystemExit(generate_mechanism_cases.main())
