from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toric_pj.experiments import fixed_kernel_2d, spectral_collision_2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 Toric PJ experiments.")
    parser.add_argument("--radius", type=int, default=24)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_args = argparse.Namespace(
        radius=args.radius,
        omega_x=0.37,
        omega_y=0.61,
        u_x=1.0,
        u_y=-0.6,
        ridge=1e-10,
        device=args.device,
        output_dir=str(output_dir),
    )
    collision_args = argparse.Namespace(
        radius=args.radius,
        omega_x=0.37,
        omega_y=0.61,
        v_x=0.8,
        v_y=-0.4,
        eps_start=-1.0,
        eps_end=-6.0,
        num_eps=13,
        ridge=1e-12,
        device=args.device,
        output_dir=str(output_dir),
    )

    result = {
        "fixed_kernel": fixed_kernel_2d.run(fixed_args),
        "spectral_collision": spectral_collision_2d.run(collision_args),
    }
    summary_path = output_dir / "stage1_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **result}, indent=2))


if __name__ == "__main__":
    main()
