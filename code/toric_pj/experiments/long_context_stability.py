from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from toric_pj.diagnostics.basis_projection import default_device


@dataclass(frozen=True)
class Variant:
    name: str
    order: int
    kind: str


VARIANTS = [
    Variant("raw_FJ_r1", 1, "raw"),
    Variant("raw_FJ_r2", 2, "raw"),
    Variant("raw_FJ_r3", 3, "raw"),
    Variant("damped_FJ_r2", 2, "damped"),
    Variant("scaled_FJ_r2", 2, "scaled"),
    Variant("LC_FJ_r2", 2, "lc"),
    Variant("LC_wrong_scale_r2", 2, "lc_wrong"),
    Variant("NTK_flow_FJ_r2", 2, "ntk_flow"),
    Variant("LC_plus_affine", 2, "lc_affine"),
]


def evaluate_variant(
    variant: Variant,
    d: torch.Tensor,
    *,
    train_length: int,
    eval_length: int,
    omega: float,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    L = float(train_length)
    W = float(eval_length)
    x = d / L
    if variant.kind == "raw":
        phase = omega * d
        bias = x.pow(variant.order) * torch.cos(phase)
    elif variant.kind == "damped":
        phase = omega * d
        bias = x.pow(variant.order) * torch.exp(-gamma * x) * torch.cos(phase)
    elif variant.kind == "scaled":
        phase = omega * d
        bias = (d / max(W, 1.0)).pow(variant.order) * torch.cos(phase)
    elif variant.kind == "lc":
        phi = L * torch.asinh(d / L)
        beta = d / torch.sqrt(d.square() + L**2)
        phase = omega * phi
        bias = beta.pow(variant.order) * torch.cos(phase)
    elif variant.kind == "lc_wrong":
        wrong_L = L / 8.0
        phi = wrong_L * torch.asinh(d / wrong_L)
        beta = d / torch.sqrt(d.square() + wrong_L**2)
        phase = omega * phi
        bias = beta.pow(variant.order) * torch.cos(phase)
    elif variant.kind == "ntk_flow":
        rho = max(W / L, 1.0)
        phase = (omega / rho) * d
        bias = x.pow(variant.order) * torch.cos(phase)
    elif variant.kind == "lc_affine":
        phi = L * torch.asinh(d / L)
        beta = d / torch.sqrt(d.square() + L**2)
        phase = omega * phi
        bias = beta.pow(variant.order) * torch.cos(phase) - 0.25 * d / W
    else:
        raise ValueError(f"unknown variant kind: {variant.kind}")
    return bias, phase


def metrics_for_variant(
    variant: Variant,
    *,
    train_length: int,
    eval_length: int,
    omega: float,
    gamma: float,
    device: torch.device,
) -> dict[str, object]:
    d = torch.arange(0, eval_length + 1, device=device, dtype=torch.float64)
    bias, phase = evaluate_variant(
        variant,
        d,
        train_length=train_length,
        eval_length=eval_length,
        omega=omega,
        gamma=gamma,
    )
    centered = bias - bias.mean()
    probs = torch.softmax(bias, dim=0)
    entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-30)))
    effective_support = torch.exp(entropy)
    far_start = max(0, int(0.9 * eval_length))
    phase_diff = torch.diff(phase)
    far_phase_step = torch.mean(torch.abs(phase_diff[far_start:])).clamp_min(0)
    raw_phase_span = omega * eval_length
    phase_span = torch.max(phase) - torch.min(phase)
    return {
        "variant": variant.name,
        "kind": variant.kind,
        "order": variant.order,
        "train_length": train_length,
        "eval_length": eval_length,
        "eval_ratio": eval_length / train_length,
        "bias_rms": float(torch.sqrt(torch.mean(bias.square())).detach().cpu()),
        "bias_std": float(torch.std(bias).detach().cpu()),
        "bias_max_abs": float(torch.max(torch.abs(bias)).detach().cpu()),
        "logit_std": float(torch.std(centered).detach().cpu()),
        "effective_support": float(effective_support.detach().cpu()),
        "effective_support_frac": float((effective_support / (eval_length + 1)).detach().cpu()),
        "phase_span": float(phase_span.detach().cpu()),
        "phase_ratio_vs_raw": float((phase_span / max(raw_phase_span, 1e-30)).detach().cpu()),
        "far_phase_step": float(far_phase_step.detach().cpu()),
        "phase_cycles": float((phase_span / (2.0 * np.pi)).detach().cpu()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_lengths = [args.train_length * ratio for ratio in args.eval_ratios]
    rows = [
        metrics_for_variant(
            variant,
            train_length=args.train_length,
            eval_length=int(eval_length),
            omega=args.omega,
            gamma=args.gamma,
            device=device,
        )
        for eval_length in eval_lengths
        for variant in VARIANTS
    ]

    csv_path = output_dir / "long_context_stability_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_scale = output_dir / "long_context_scale_metrics.png"
    plot_phase = output_dir / "long_context_phase_metrics.png"
    plot_metrics(rows, plot_scale, plot_phase)

    summary = {
        "device": str(device),
        "train_length": args.train_length,
        "eval_ratios": args.eval_ratios,
        "csv": str(csv_path),
        "scale_plot": str(plot_scale),
        "phase_plot": str(plot_phase),
        "rows": rows,
    }
    summary_path = output_dir / "long_context_stability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def plot_metrics(rows: list[dict[str, object]], scale_path: Path, phase_path: Path) -> None:
    variants = [variant.name for variant in VARIANTS]
    ratios = sorted({float(row["eval_ratio"]) for row in rows})
    by_variant = {variant: [row for row in rows if row["variant"] == variant] for variant in variants}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for variant in variants:
        values = sorted(by_variant[variant], key=lambda row: float(row["eval_ratio"]))
        x = [float(row["eval_ratio"]) for row in values]
        axes[0].loglog(x, [float(row["bias_max_abs"]) for row in values], marker="o", label=variant)
        axes[1].loglog(x, [float(row["bias_std"]) for row in values], marker="o")
        axes[2].semilogx(x, [float(row["effective_support_frac"]) for row in values], marker="o")
    axes[0].set_title("Bias max abs")
    axes[1].set_title("Logit std")
    axes[2].set_title("Effective support fraction")
    for ax in axes:
        ax.set_xlabel("eval/train length")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(scale_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for variant in variants:
        values = sorted(by_variant[variant], key=lambda row: float(row["eval_ratio"]))
        x = [float(row["eval_ratio"]) for row in values]
        axes[0].semilogx(x, [float(row["phase_ratio_vs_raw"]) for row in values], marker="o", label=variant)
        axes[1].loglog(x, [max(float(row["far_phase_step"]), 1e-12) for row in values], marker="o")
    axes[0].set_title("Phase span / raw phase span")
    axes[1].set_title("Far-field phase step")
    for ax in axes:
        ax.set_xlabel("eval/train length")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(phase_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scalar-bias long-context stability diagnostics.")
    parser.add_argument("--train-length", type=int, default=256)
    parser.add_argument("--eval-ratios", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--omega", type=float, default=0.035)
    parser.add_argument("--gamma", type=float, default=0.45)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stageG")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
