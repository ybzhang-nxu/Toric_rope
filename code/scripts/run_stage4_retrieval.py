from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toric_pj.experiments import trainable_attention_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4 trainable attention retrieval bridge.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage4_retrieval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = trainable_attention_retrieval.run(
        argparse.Namespace(
            side=12,
            steps=900,
            lr=0.08,
            coeff_l2=1e-7,
            vocab_size=64,
            batch_size=128,
            eval_batches=16,
            seed=909,
            device=args.device,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
