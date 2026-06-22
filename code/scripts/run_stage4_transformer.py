from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toric_pj.experiments import mini_transformer_copy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4 mini Transformer copy tasks.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage4_transformer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = mini_transformer_copy.run(
        argparse.Namespace(
            side=10,
            steps=900,
            batch_size=128,
            eval_batches=12,
            vocab_size=64,
            dim=64,
            n_heads=2,
            lr=0.012,
            coeff_l2=1e-7,
            seed=1701,
            device=args.device,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
