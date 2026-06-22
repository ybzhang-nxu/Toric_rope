from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toric_pj.experiments import real_digits_transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2-E real digits Transformer benchmark.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v2_real_digits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = real_digits_transformer.run(
        argparse.Namespace(
            side=8,
            train_count=1400,
            dim=48,
            n_heads=2,
            batch_size=128,
            eval_batch_size=256,
            recon_steps=260,
            cls_steps=320,
            mask_rate=0.35,
            lr=0.003,
            weight_decay=1e-4,
            seed=515,
            device=args.device,
            output_dir=args.output_dir,
        )
    )
    printable = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
