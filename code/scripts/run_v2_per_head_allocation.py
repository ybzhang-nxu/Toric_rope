from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toric_pj.experiments import per_head_allocation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2 per-head branch allocation.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v2_per_head_allocation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = per_head_allocation.run(
        argparse.Namespace(
            radius=18,
            n_heads=4,
            steps=900,
            lr=0.035,
            gate_l1=1e-4,
            coeff_l2=1e-7,
            entropy_weight=2e-3,
            balance_weight=2e-2,
            seed=606,
            device=args.device,
            output_dir=args.output_dir,
        )
    )
    printable = {key: value for key, value in result.items() if key not in {"gate_rows", "histories"}}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
