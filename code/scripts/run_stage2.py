from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toric_pj.experiments import adaptive_teacher_2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 adaptive teacher experiments.")
    parser.add_argument("--radius", type=int, default=24)
    parser.add_argument("--steps", type=int, default=3500)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = adaptive_teacher_2d.run(
        argparse.Namespace(
            radius=args.radius,
            steps=args.steps,
            lr=0.03,
            gate_l1=2e-5,
            coeff_l2=1e-7,
            restarts=args.restarts,
            seed=17,
            device=args.device,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

