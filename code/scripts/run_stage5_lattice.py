from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toric_pj.experiments import lattice_physics_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 5 lattice physics probe.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage5_lattice")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = lattice_physics_probe.run(
        argparse.Namespace(
            side=14,
            ridge=1e-9,
            batch_size=64,
            eval_batches=8,
            seed=909,
            device=args.device,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
