from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from toric_pj.diagnostics.basis_projection import Basis, normalize_columns
from toric_pj.diagnostics.relative_table_geometry import (
    axial_projection,
    axis_lowrank_basis,
    dct2,
    fit_linear_basis,
    fixed_toric_pj_basis,
    fourier_atom_basis,
    full_axial_basis,
    load_bias_npz,
    random_matched_radius_omegas,
    relative_grid,
    table_informed_pj_basis,
    top_dft_omegas,
    write_csv,
)
from toric_pj.experiments.v3_digits_transformer_scaling import PRUNED_REAL_DIGITS_GROUPS, label_group, prune_basis


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def top_dct_indices(table: torch.Tensor, *, k: int) -> torch.Tensor:
    coeff = dct2(table - table.mean()).square()
    coeff[0, 0] = 0.0
    vals, idx = torch.topk(coeff.reshape(-1), k=min(k, coeff.numel()))
    del vals
    return torch.stack([idx // table.shape[-1], idx % table.shape[-1]], dim=1)


def top_dct_basis(table: torch.Tensor, side: int, *, budget: int, name: str) -> Basis:
    size = table.shape[-1]
    k = max(1, budget)
    indices = top_dct_indices(table, k=k)
    d = relative_grid(side, device=table.device, dtype=table.dtype)
    x = (d[:, 0] + side - 1).reshape(-1)
    y = (d[:, 1] + side - 1).reshape(-1)
    cols = [torch.ones_like(x)]
    labels = ["const"]
    for rank, (kx, ky) in enumerate(indices.tolist(), start=1):
        col = torch.cos(math.pi / float(size) * (x + 0.5) * int(kx)) * torch.cos(
            math.pi / float(size) * (y + 0.5) * int(ky)
        )
        cols.append(col)
        labels.append(f"dct{rank}_kx{int(kx)}_ky{int(ky)}")
    return Basis(name, torch.stack(cols, dim=1), labels, [0] * len(labels))


def basis_for_variant(variant: str, target: torch.Tensor, side: int, budget: int, seed: int) -> Basis | None:
    device = target.device
    dtype = target.dtype
    if variant == "axis_lowrank_5":
        return axis_lowrank_basis(side, device=device, dtype=dtype)
    if variant == "axis_full_29":
        return full_axial_basis(side, device=device, dtype=dtype)
    if variant in {"topk_dft", "windowed_topk_dft", "random_spectral_atoms_matched_radius"}:
        k = max(1, (budget - 1) // 2)
        if variant == "topk_dft":
            omegas = top_dft_omegas(target, k=k, window=False)
        elif variant == "windowed_topk_dft":
            omegas = top_dft_omegas(target, k=k, window=True)
        else:
            omegas = random_matched_radius_omegas(target, k=k, seed=seed)
        return fourier_atom_basis(side, omegas, name=f"{variant}_{budget}")
    if variant == "topk_dct":
        return top_dct_basis(target, side, budget=budget, name=f"{variant}_{budget}")
    if variant == "fixed_toric_PJ_R2":
        return fixed_toric_pj_basis(side, device=device, dtype=dtype)
    if variant == "pruned_toric_PJ":
        return prune_basis(fixed_toric_pj_basis(side, device=device, dtype=dtype), PRUNED_REAL_DIGITS_GROUPS, name="pruned_toric_PJ")
    if variant.startswith("table_informed_toric_PJ_R"):
        order = int(variant.rsplit("R", 1)[1])
        # Directional PJ creates many columns per atom, so choose K conservatively.
        per_atom = {0: 2, 1: 10, 2: 18}.get(order, 18)
        k = max(1, (budget - 1) // per_atom)
        omegas = top_dft_omegas(target, k=k)
        return table_informed_pj_basis(side, omegas, order=order, name=f"{variant}_{budget}")
    if variant.startswith("axis_plus_toric_residual_R"):
        order = int(variant.rsplit("R", 1)[1])
        per_atom = {0: 2, 1: 10, 2: 18}.get(order, 18)
        k = max(1, (budget - 29) // per_atom)
        omegas = top_dft_omegas(target, k=k)
        return table_informed_pj_basis(side, omegas, order=order, name=f"{variant}_{budget}")
    if variant == "axis_plus_lc_toric_residual":
        # First implementation uses the same residual PJ atoms; LC/log coordinate comes in the train-time variant.
        k = max(1, (budget - 29) // 18)
        omegas = top_dft_omegas(target, k=k)
        return table_informed_pj_basis(side, omegas, order=2, name=f"{variant}_{budget}")
    if variant in {"continuous_bias_mlp_raw", "continuous_bias_mlp_logcoord"}:
        return None
    raise ValueError(f"unknown projection variant: {variant}")


class TinyBiasMLP(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_mlp(table: torch.Tensor, side: int, *, logcoord: bool, steps: int, lr: float, seed: int) -> tuple[torch.Tensor, float, float]:
    torch.manual_seed(seed)
    d = relative_grid(side, device=table.device, dtype=table.dtype)
    x = d / float(max(1, side - 1))
    if logcoord:
        x = torch.sign(x) * torch.log1p(torch.abs(x))
    target = table.reshape(-1)
    model = TinyBiasMLP(hidden=32).to(table.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(x.float()).to(table.dtype)
        loss = torch.mean((pred - target).square())
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(x.float()).to(table.dtype).reshape_as(table)
        mse = torch.mean((pred - table).square())
        var = torch.mean((table - table.mean()).square()).clamp_min(1e-12)
        r2 = 1.0 - mse / var
    return pred, float(mse.detach().cpu()), float(r2.detach().cpu())


def fit_projection(
    *,
    variant: str,
    fit_target: str,
    table: torch.Tensor,
    side: int,
    budget: int,
    seed: int,
    mlp_steps: int,
) -> dict[str, object]:
    axis = axial_projection(table)
    residual = table - axis
    if fit_target == "full_table":
        target = table
        base = torch.zeros_like(table)
    elif fit_target == "oblique_residual":
        target = residual
        base = torch.zeros_like(table)
    elif fit_target == "axis_plus_residual":
        target = residual
        base = axis
    else:
        raise ValueError(f"unknown fit target: {fit_target}")

    if variant in {"continuous_bias_mlp_raw", "continuous_bias_mlp_logcoord"}:
        pred_residual, mse, r2 = fit_mlp(
            target,
            side,
            logcoord=variant.endswith("logcoord"),
            steps=mlp_steps,
            lr=2e-3,
            seed=seed,
        )
        num_features = sum(param.numel() for param in TinyBiasMLP(hidden=32).parameters())
        coeff_norm = float("nan")
        basis_name = variant
    else:
        basis = basis_for_variant(variant, target, side, budget, seed)
        assert basis is not None
        pred_residual, coeff, mse, r2 = fit_linear_basis(basis.matrix.to(table.device, table.dtype), target)
        num_features = int(basis.matrix.shape[1])
        coeff_norm = float(torch.linalg.norm(coeff).detach().cpu())
        basis_name = basis.name

    pred_full = base + pred_residual
    full_mse = torch.mean((pred_full - table).square())
    full_var = torch.mean((table - table.mean()).square()).clamp_min(1e-12)
    full_r2 = 1.0 - full_mse / full_var
    if fit_target == "full_table":
        residual_r2_value = float("nan")
    else:
        residual_var = torch.mean((residual - residual.mean()).square()).clamp_min(1e-12)
        residual_mse = torch.mean((pred_residual - residual).square())
        residual_r2 = 1.0 - residual_mse / residual_var
        residual_r2_value = float(residual_r2.detach().cpu())
    return {
        "variant": variant,
        "basis_name": basis_name,
        "fit_target": fit_target,
        "feature_budget": budget,
        "num_features": num_features,
        "effective_params": num_features,
        "bias_mse": mse,
        "table_fit_r2": float(full_r2.detach().cpu()),
        "target_fit_r2": r2,
        "residual_fit_r2": residual_r2_value,
        "axis_plus_residual_fit_r2": float(full_r2.detach().cpu()) if fit_target == "axis_plus_residual" else float("nan"),
        "coeff_norm": coeff_norm,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    bundle, metadata = load_bias_npz(Path(args.input), device=device)
    variants = parse_list(args.variants)
    budgets = parse_int_list(args.feature_budgets)
    fit_targets = parse_list(args.fit_targets)
    side = int(args.grid_side or ((bundle.tables.shape[-1] + 1) // 2))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    for layer in range(bundle.tables.shape[0]):
        for head in range(bundle.tables.shape[1]):
            table = bundle.tables[layer, head]
            for fit_target in fit_targets:
                for budget in budgets:
                    for variant in variants:
                        row = fit_projection(
                            variant=variant,
                            fit_target=fit_target,
                            table=table,
                            side=side,
                            budget=budget,
                            seed=args.seed + 1000 * layer + 17 * head + budget,
                            mlp_steps=args.mlp_steps,
                        )
                        row.update(
                            {
                                "dataset": metadata.get("dataset", "unknown"),
                                "task": metadata.get("task", "unknown"),
                                "teacher_basis": metadata.get("basis", "unknown"),
                                "teacher_seed": metadata.get("seed", -1),
                                "layer": layer,
                                "head": head,
                            }
                        )
                        rows.append(row)
                        if "residual" in fit_target or "axis_plus" in variant:
                            residual_rows.append(row)
    write_csv(output_dir / "projection_results.csv", rows)
    write_csv(output_dir / "residual_projection_results.csv", residual_rows)
    random_rows = [row for row in rows if row["variant"] == "random_spectral_atoms_matched_radius"]
    write_csv(output_dir / "random_matched_radius_controls.csv", random_rows)

    groups: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["variant"]), str(row["fit_target"]), int(row["feature_budget"])), []).append(row)
    aggregate: list[dict[str, object]] = []
    for (variant, fit_target, budget), values in sorted(groups.items()):
        table_r2 = torch.tensor([float(row["table_fit_r2"]) for row in values])
        residual_r2 = torch.tensor([float(row["residual_fit_r2"]) for row in values])
        aggregate.append(
            {
                "variant": variant,
                "fit_target": fit_target,
                "feature_budget": budget,
                "n": len(values),
                "table_fit_r2_mean": float(table_r2.mean()),
                "table_fit_r2_std": float(table_r2.std(unbiased=False)),
                "residual_fit_r2_mean": float(residual_r2.mean()),
                "residual_fit_r2_std": float(residual_r2.std(unbiased=False)),
                "num_features": int(values[0]["num_features"]),
            }
        )
    write_csv(output_dir / "projection_aggregate.csv", aggregate)
    write_csv(output_dir / "curvature_fit_results.csv", [])
    write_csv(output_dir / "topk_atom_tables.csv", [])
    write_report(output_dir, metadata, aggregate)
    summary = {
        "input": args.input,
        "output_dir": str(output_dir),
        "num_rows": len(rows),
        "num_aggregate_rows": len(aggregate),
    }
    (output_dir / "table_projection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_report(output_dir: Path, metadata: dict[str, object], aggregate: list[dict[str, object]]) -> None:
    lines = [
        "# V4 Table Projection Report",
        "",
        "Teacher:",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| variant | target | budget | features | n | table R2 | residual R2 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate:
        lines.append(
            "| "
            + f"{row['variant']} | {row['fit_target']} | {int(row['feature_budget'])} | "
            + f"{int(row['num_features'])} | {int(row['n'])} | "
            + f"{float(row['table_fit_r2_mean']):.4f} | {float(row['residual_fit_r2_mean']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `axis_plus_residual` fits the oblique residual and adds the best full axial projection back.",
            "- `random_spectral_atoms_matched_radius` controls for feature budget and radial frequency profile.",
            "- DFT variants use cos/sin atoms; DCT variants are non-periodic boundary controls.",
            "",
            "Artifacts:",
            "",
            "- `projection_results.csv`",
            "- `projection_aggregate.csv`",
            "- `residual_projection_results.csv`",
            "- `random_matched_radius_controls.csv`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V4 residual-first table projection.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--grid-side", type=int, default=None)
    parser.add_argument("--feature-budgets", type=str, default="5,15,33,55")
    parser.add_argument("--fit-targets", type=str, default="full_table,oblique_residual,axis_plus_residual")
    parser.add_argument(
        "--variants",
        type=str,
        default="axis_lowrank_5,axis_full_29,topk_dft,topk_dct,windowed_topk_dft,random_spectral_atoms_matched_radius,fixed_toric_PJ_R2,pruned_toric_PJ,table_informed_toric_PJ_R0,table_informed_toric_PJ_R1,table_informed_toric_PJ_R2,axis_plus_toric_residual_R0,axis_plus_toric_residual_R1,axis_plus_toric_residual_R2",
    )
    parser.add_argument("--mlp-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results/v4_table_projection_cifar10")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
