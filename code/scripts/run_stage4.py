from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toric_pj.experiments import synthetic_attention_2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4 synthetic attention bridge experiments.")
    parser.add_argument("--side", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage4")
    parser.add_argument("--learn-geometry", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = synthetic_attention_2d.run(
        argparse.Namespace(
            side=args.side,
            steps=args.steps,
            lr=0.035,
            teacher_temperature=1.0,
            gate_l1=1e-5,
            freeze_geometry=not args.learn_geometry,
            device=args.device,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
