from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toric_pj.experiments import long_context_stability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage G long-context stability diagnostics.")
    parser.add_argument("--train-length", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stageG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = long_context_stability.run(
        argparse.Namespace(
            train_length=args.train_length,
            eval_ratios=[1, 4, 16, 32],
            omega=0.035,
            gamma=0.45,
            device=args.device,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
