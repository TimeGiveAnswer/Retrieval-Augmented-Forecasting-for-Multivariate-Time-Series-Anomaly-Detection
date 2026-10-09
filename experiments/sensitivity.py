"""Chapter 3.7: K, memory size, window length and score weight sweeps."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import run_extended_ablations

if __name__ == "__main__":
    tokens = sys.argv[1:]
    if tokens and tokens[0] in {"run", "plan"} and "--suites" not in tokens:
        tokens.extend(["--suites", "sensitivity"])
    raise SystemExit(run_extended_ablations.main(tokens))
