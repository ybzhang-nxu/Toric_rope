from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toric_pj.experiments import ablation_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage ablation diagnostics.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = ablation_suite.run(
        argparse.Namespace(
            train_radius=32,
            eval_radius=256,
            spectral_train_radius=20,
            spectral_eval_radius=40,
            lc_train_radius=64,
            lc_eval_radius=512,
            gamma=0.85,
            ridge=1e-9,
            device=args.device,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
