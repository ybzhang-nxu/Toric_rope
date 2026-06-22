from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toric_pj.experiments import learnable_spectral_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2-A learnable spectral geometry.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v2_spectral")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = learnable_spectral_geometry.run(
        argparse.Namespace(
            train_radius=12,
            eval_radius=20,
            max_order=2,
            steps=420,
            lr_coeff=0.045,
            lr_geom=0.018,
            coeff_l2=1e-7,
            lc_scale_reg=1e-5,
            ridge=1e-9,
            seeds=[101, 202],
            device=args.device,
            output_dir=args.output_dir,
        )
    )
    printable = {key: value for key, value in result.items() if key not in {"rows", "histories"}}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
