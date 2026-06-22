from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toric_pj.experiments import language_hierarchical_synthetic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 5 hierarchical language coordinate probe.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage5_language")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = language_hierarchical_synthetic.run(
        argparse.Namespace(
            num_tokens=4096,
            segment_size=128,
            max_lag=512,
            samples_per_lag=256,
            ridge=1e-9,
            seed=321,
            device=args.device,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
