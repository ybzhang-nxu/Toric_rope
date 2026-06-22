from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from toric_pj.diagnostics.basis_projection import default_device
from toric_pj.diagnostics.direction_alignment import normalize_direction
from toric_pj.models.toric_pj_rotary import exact_toric_pj_action, semisimple_character


def semisimple_error(device: torch.device, *, samples: int, radius: int) -> float:
    torch.manual_seed(123)
    omega = torch.tensor([0.37, 0.61], device=device, dtype=torch.float64)
    q = torch.randn(samples, device=device, dtype=torch.complex128)
    k = torch.randn(samples, device=device, dtype=torch.complex128)
    p_i = torch.randint(-radius, radius + 1, (samples, 2), device=device)
    p_j = torch.randint(-radius, radius + 1, (samples, 2), device=device)
    direct = []
    expected = []
    for idx in range(samples):
        zi = semisimple_character(p_i[idx], omega)
        zj = semisimple_character(p_j[idx], omega)
        direct.append((zi * q[idx]) * torch.conj(zj * k[idx]))
        expected.append(q[idx] * torch.conj(k[idx]) * semisimple_character(p_i[idx] - p_j[idx], omega))
    direct_t = torch.stack(direct)
    expected_t = torch.stack(expected)
    return float(torch.max(torch.abs(direct_t - expected_t)).detach().cpu())


def first_order_jet_error(device: torch.device, *, radius: int) -> float:
    omega = torch.tensor([0.37, 0.61], device=device, dtype=torch.float64)
    direction = normalize_direction(torch.tensor([[1.0, -0.6]], device=device, dtype=torch.float64)).reshape(-1)
    errors = []
    for dx in range(0, radius + 1):
        for dy in range(0, radius + 1):
            d = torch.tensor([dx, dy], device=device, dtype=torch.long)
            action = exact_toric_pj_action(d, omega, direction, order=1)
            phase = semisimple_character(d, omega)
            expected = phase * torch.sum(direction * d.to(torch.float64))
            errors.append(torch.abs(action[0, 1] - expected))
    return float(torch.max(torch.stack(errors)).detach().cpu())


def norm_curve(
    device: torch.device,
    *,
    train_length: int,
    eval_ratios: list[int],
    omega: torch.Tensor,
    direction: torch.Tensor,
) -> list[dict[str, object]]:
    rows = []
    for ratio in eval_ratios:
        eval_length = train_length * ratio
        for order in [0, 1, 2, 3]:
            d = torch.tensor([eval_length, eval_length // 4], device=device, dtype=torch.long)
            if order == 0:
                norm = 1.0
                fro = 1.0
            else:
                action = exact_toric_pj_action(d, omega, direction, order=order, scale=1.0)
                norm = float(torch.linalg.svdvals(action)[0].detach().cpu())
                fro = float(torch.linalg.matrix_norm(action, ord="fro").detach().cpu())
            rows.append(
                {
                    "variant": f"exact_order{order}",
                    "order": order,
                    "eval_ratio": ratio,
                    "eval_length": eval_length,
                    "operator_norm": norm,
                    "fro_norm": fro,
                }
            )
        for order in [1, 2, 3]:
            d = torch.tensor([eval_length, eval_length // 4], device=device, dtype=torch.long)
            action = exact_toric_pj_action(d, omega, direction, order=order, scale=float(eval_length))
            rows.append(
                {
                    "variant": f"scaled_exact_order{order}",
                    "order": order,
                    "eval_ratio": ratio,
                    "eval_length": eval_length,
                    "operator_norm": float(torch.linalg.svdvals(action)[0].detach().cpu()),
                    "fro_norm": float(torch.linalg.matrix_norm(action, ord="fro").detach().cpu()),
                }
            )
    return rows


def complex_normal(shape: tuple[int, ...], *, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    real = torch.randn(shape, device=device, dtype=torch.float64, generator=generator)
    imag = torch.randn(shape, device=device, dtype=torch.float64, generator=generator)
    return torch.complex(real, imag) / (2.0 * float(shape[-1])) ** 0.5


def lattice_path_indices(length: int, *, samples: int, device: torch.device) -> torch.Tensor:
    count = min(length, samples)
    idx = torch.linspace(0, length - 1, steps=count, device=device, dtype=torch.float64).round().to(torch.long)
    return torch.unique(idx)


def action_for_position(
    index: int,
    *,
    omega: torch.Tensor,
    direction: torch.Tensor,
    order: int,
    scale: float,
) -> torch.Tensor:
    device = omega.device
    position = torch.tensor([index, index // 4], device=device, dtype=torch.long)
    if order == 0:
        return semisimple_character(position, omega).reshape(1, 1).to(torch.complex128)
    return exact_toric_pj_action(position, omega, direction, order=order, scale=scale)


def transform_vectors(
    vectors: torch.Tensor,
    indices: torch.Tensor,
    *,
    omega: torch.Tensor,
    direction: torch.Tensor,
    order: int,
    scale: float,
) -> torch.Tensor:
    output = torch.empty_like(vectors)
    cache: dict[int, torch.Tensor] = {}
    for row, index in enumerate(indices.detach().cpu().tolist()):
        index = int(index)
        action = cache.get(index)
        if action is None:
            action = action_for_position(index, omega=omega, direction=direction, order=order, scale=scale)
            cache[index] = action
        output[row] = action @ vectors[row]
    return output


def feature_transform_metrics(
    device: torch.device,
    *,
    train_length: int,
    eval_ratios: list[int],
    omega: torch.Tensor,
    direction: torch.Tensor,
    cache_samples: int,
    logit_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    variants = [{"variant": "semisimple_order0", "order": 0, "scale_mode": "unit"}]
    for order in [1, 2, 3]:
        variants.extend(
            [
                {"variant": f"raw_exact_order{order}", "order": order, "scale_mode": "raw"},
                {"variant": f"train_scaled_order{order}", "order": order, "scale_mode": "train_length"},
                {"variant": f"eval_scaled_order{order}", "order": order, "scale_mode": "eval_length"},
            ]
        )

    for ratio in eval_ratios:
        eval_length = train_length * ratio
        for variant in variants:
            order = int(variant["order"])
            scale_mode = str(variant["scale_mode"])
            if scale_mode == "raw" or order == 0:
                scale = 1.0
            elif scale_mode == "train_length":
                scale = float(train_length)
            elif scale_mode == "eval_length":
                scale = float(eval_length)
            else:
                raise ValueError(f"unknown scale mode: {scale_mode}")

            dim = order + 1
            cache_idx = lattice_path_indices(eval_length, samples=cache_samples, device=device)
            keys = complex_normal((cache_idx.numel(), dim), device=device, generator=gen)
            transformed_keys = transform_vectors(
                keys,
                cache_idx,
                omega=omega,
                direction=direction,
                order=order,
                scale=scale,
            )
            key_norm = torch.linalg.norm(transformed_keys, dim=-1)

            q_idx = torch.randint(0, eval_length, (logit_samples,), device=device, generator=gen)
            k_idx = torch.randint(0, eval_length, (logit_samples,), device=device, generator=gen)
            q_base = complex_normal((logit_samples, dim), device=device, generator=gen)
            k_base = complex_normal((logit_samples, dim), device=device, generator=gen)
            q = transform_vectors(q_base, q_idx, omega=omega, direction=direction, order=order, scale=scale)
            k = transform_vectors(k_base, k_idx, omega=omega, direction=direction, order=order, scale=scale)
            scores = torch.real(torch.sum(q * torch.conj(k), dim=-1)) / float(dim) ** 0.5

            rows.append(
                {
                    "variant": str(variant["variant"]),
                    "order": order,
                    "scale_mode": scale_mode,
                    "eval_ratio": ratio,
                    "eval_length": eval_length,
                    "scale": scale,
                    "key_norm_mean": float(torch.mean(key_norm).detach().cpu()),
                    "key_norm_p95": float(torch.quantile(key_norm, 0.95).detach().cpu()),
                    "key_norm_max": float(torch.max(key_norm).detach().cpu()),
                    "cache_rms": float(torch.sqrt(torch.mean(key_norm.square())).detach().cpu()),
                    "logit_std": float(torch.std(scores).detach().cpu()),
                    "logit_abs_p95": float(torch.quantile(torch.abs(scores), 0.95).detach().cpu()),
                }
            )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    omega = torch.tensor([0.37, 0.61], device=device, dtype=torch.float64)
    direction = normalize_direction(torch.tensor([[1.0, -0.6]], device=device, dtype=torch.float64)).reshape(-1)

    sanity = {
        "semisimple_relative_phase_max_error": semisimple_error(device, samples=args.samples, radius=args.radius),
        "first_order_jet_max_error": first_order_jet_error(device, radius=args.radius),
    }
    rows = norm_curve(
        device,
        train_length=args.train_length,
        eval_ratios=args.eval_ratios,
        omega=omega,
        direction=direction,
    )
    csv_path = output_dir / "exact_rotary_norms.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = output_dir / "exact_rotary_norms.png"
    plot_norms(rows, plot_path)
    feature_rows = feature_transform_metrics(
        device,
        train_length=args.train_length,
        eval_ratios=args.eval_ratios,
        omega=omega,
        direction=direction,
        cache_samples=args.cache_samples,
        logit_samples=args.logit_samples,
        seed=args.seed,
    )
    feature_csv_path = output_dir / "exact_rotary_feature_metrics.csv"
    with feature_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)
    feature_plot_path = output_dir / "exact_rotary_feature_metrics.png"
    plot_feature_metrics(feature_rows, feature_plot_path)
    summary = {
        "device": str(device),
        "sanity": sanity,
        "csv": str(csv_path),
        "plot": str(plot_path),
        "rows": rows,
        "feature_csv": str(feature_csv_path),
        "feature_plot": str(feature_plot_path),
        "feature_rows": feature_rows,
    }
    summary_path = output_dir / "exact_rotary_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def plot_norms(rows: list[dict[str, object]], path: Path) -> None:
    variants = sorted({str(row["variant"]) for row in rows})
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant in variants:
        values = sorted([row for row in rows if row["variant"] == variant], key=lambda item: float(item["eval_ratio"]))
        ax.loglog(
            [float(row["eval_ratio"]) for row in values],
            [float(row["operator_norm"]) for row in values],
            marker="o",
            label=variant,
        )
    ax.set_xlabel("eval/train length")
    ax.set_ylabel("operator norm")
    ax.set_title("Exact Toric PJ-Rotary Feature-Action Norm")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_feature_metrics(rows: list[dict[str, object]], path: Path) -> None:
    variants = [
        "semisimple_order0",
        "raw_exact_order1",
        "raw_exact_order2",
        "raw_exact_order3",
        "train_scaled_order3",
        "eval_scaled_order3",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, title in [
        (axes[0], "key_norm_p95", "Key Norm p95 / Cache Scale Proxy"),
        (axes[1], "logit_abs_p95", "Attention Logit |p95|"),
    ]:
        for variant in variants:
            values = sorted([row for row in rows if row["variant"] == variant], key=lambda item: float(item["eval_ratio"]))
            if not values:
                continue
            ax.loglog(
                [float(row["eval_ratio"]) for row in values],
                [float(row[metric]) for row in values],
                marker="o",
                label=variant,
            )
        ax.set_xlabel("eval/train length")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("value")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact toric PJ-rotary sanity and stability checks.")
    parser.add_argument("--radius", type=int, default=32)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--train-length", type=int, default=256)
    parser.add_argument("--eval-ratios", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--cache-samples", type=int, default=384)
    parser.add_argument("--logit-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage_rotary")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
