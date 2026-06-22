from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toric_pj.experiments import pj_ntk_diagnostic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 projected PJ-NTK diagnostics.")
    parser.add_argument("--radius", type=int, default=12)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage3")
    parser.add_argument("--stage2-summary", type=str, default="results/stage2/adaptive_teacher_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = pj_ntk_diagnostic.run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
