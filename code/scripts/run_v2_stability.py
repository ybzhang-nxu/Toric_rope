from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toric_pj.experiments import stability_constraints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2-C stability constraints.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v2_stability")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = stability_constraints.run(
        argparse.Namespace(
            damping_train_radius=28,
            damping_calib_radius=56,
            damping_eval_radius=160,
            lc_train_radius=48,
            lc_calib_radius=96,
            lc_eval_radius=160,
            gamma=0.85,
            steps=650,
            lr=0.035,
            coeff_l2=1e-7,
            ridge=1e-9,
            device=args.device,
            output_dir=args.output_dir,
        )
    )
    printable = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
