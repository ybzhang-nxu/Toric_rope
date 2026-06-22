from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toric_pj.experiments import sparse_pruning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2-B sparse pruning.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v2_pruning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = sparse_pruning.run(
        argparse.Namespace(
            ridge=1e-6,
            seed=909,
            device=args.device,
            output_dir=args.output_dir,
        )
    )
    printable = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
