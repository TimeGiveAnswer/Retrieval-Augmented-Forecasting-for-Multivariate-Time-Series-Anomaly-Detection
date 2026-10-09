"""Additional resources experiment: parameters, timing, memory, K/M scaling."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import measure_efficiency_resources

if __name__ == "__main__":
    raise SystemExit(measure_efficiency_resources.main())
