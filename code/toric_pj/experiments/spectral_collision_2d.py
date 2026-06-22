from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from toric_pj.diagnostics.basis_projection import default_device, make_grid_2d
from toric_pj.diagnostics.direction_alignment import normalize_direction
from toric_pj.diagnostics.spectral_collision import collision_curve


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    dtype = torch.float64
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    d = make_grid_2d(args.radius, signed=True, device=device, dtype=dtype)
    omega = torch.tensor([args.omega_x, args.omega_y], device=device, dtype=dtype)
    v = normalize_direction(torch.tensor([[args.v_x, args.v_y]], device=device, dtype=dtype)).reshape(-1)
    eps_values = [10.0 ** exponent for exponent in torch.linspace(args.eps_start, args.eps_end, args.num_eps)]
    rows = collision_curve(d, omega, v, [float(value) for value in eps_values], ridge=args.ridge)

    csv_path = output_dir / "spectral_collision_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = output_dir / "spectral_collision_condition.png"
    plot_collision(rows, plot_path)

    summary = {
        "device": str(device),
        "num_points": int(d.shape[0]),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "smallest_eps": rows[-1],
    }
    summary_path = output_dir / "spectral_collision_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def plot_collision(rows: list[dict[str, float]], path: Path) -> None:
    eps = [row["eps"] for row in rows]
    two_cond = [row["two_freq_condition"] for row in rows]
    jet_cond = [row["jet_condition"] for row in rows]
    two_mse = [row["two_freq_mse"] for row in rows]
    jet_mse = [row["jet_mse"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].loglog(eps, two_cond, marker="o", label="two close frequencies")
    axes[0].loglog(eps, jet_cond, marker="o", label="explicit first jet")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("epsilon")
    axes[0].set_ylabel("normalized basis condition")
    axes[0].set_title("Spectral collision conditioning")
    axes[0].legend()

    axes[1].loglog(eps, two_mse, marker="o", label="two close frequencies")
    axes[1].loglog(eps, jet_mse, marker="o", label="explicit first jet")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("epsilon")
    axes[1].set_ylabel("fit MSE")
    axes[1].set_title("Difference target fit")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2D spectral collision diagnostics.")
    parser.add_argument("--radius", type=int, default=24)
    parser.add_argument("--omega-x", type=float, default=0.37)
    parser.add_argument("--omega-y", type=float, default=0.61)
    parser.add_argument("--v-x", type=float, default=0.8)
    parser.add_argument("--v-y", type=float, default=-0.4)
    parser.add_argument("--eps-start", type=float, default=-1.0)
    parser.add_argument("--eps-end", type=float, default=-6.0)
    parser.add_argument("--num-eps", type=int, default=13)
    parser.add_argument("--ridge", type=float, default=1e-12)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage1")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

