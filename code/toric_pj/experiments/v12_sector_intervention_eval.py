from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from toric_pj.diagnostics.basis_projection import Basis, default_device
from toric_pj.diagnostics.relative_table_geometry import load_bias_npz, pairwise_to_relative_table
from toric_pj.experiments.v3_real_vision_scaling import (
    VisionRelPosTransformer,
    evaluate_classifier,
    evaluate_reconstruction,
    load_vision_dataset,
    normalize_patches,
    patchify,
)
from toric_pj.experiments.v4_metric_toric_pj import build_teacher_bases, parse_list
from toric_pj.experiments.v4_offset_holdout import bias_overrides_from_model, zero_bias_overrides


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def resolve_path(value: str | Path, *, cwd: Path) -> Path:
    path = Path(value)
    candidates = [
        path,
        cwd / path,
        cwd / "MetricToric" / path,
    ]
    if path.parts and path.parts[0] == "results":
        candidates.append(cwd / "MetricToric" / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def checkpoint_seed(checkpoint: dict[str, object]) -> int:
    metadata = checkpoint.get("metadata", {})
    if isinstance(metadata, dict) and "seed" in metadata:
        return int(metadata["seed"])
    args = checkpoint.get("args", {})
    if isinstance(args, dict) and "seed" in args:
        return int(args["seed"])
    return 426


def rebuild_checkpoint_basis(
    checkpoint: dict[str, object],
    *,
    teacher_bias: Path | None,
    device: torch.device,
) -> Basis:
    basis_name = str(checkpoint.get("basis_name", checkpoint.get("metadata", {}).get("basis", "checkpoint_basis")))
    basis_matrix = checkpoint["basis_matrix"].to(device=device, dtype=torch.float32)
    orders = list(checkpoint.get("basis_orders", [0] * basis_matrix.shape[1]))
    generic = Basis(basis_name, basis_matrix, [f"f{i}" for i in range(basis_matrix.shape[1])], orders)
    if teacher_bias is None:
        return generic

    try:
        bundle, metadata = load_bias_npz(teacher_bias, device=torch.device("cpu"))
        side = int(checkpoint["metadata"].get("grid_side", metadata.get("grid_side", 8)))
        rebuilt = build_teacher_bases(
            side=side,
            device=device,
            teacher_tables=bundle.tables.float().to(device),
            variants=[basis_name],
            include_shuffle=False,
            seed=checkpoint_seed(checkpoint),
        )[0]
    except Exception:
        return generic

    if rebuilt.matrix.shape[1] != basis_matrix.shape[1]:
        return generic
    return Basis(basis_name, rebuilt.matrix.to(device=device, dtype=torch.float32), rebuilt.labels, rebuilt.orders)


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    state: str,
    teacher_bias: Path | None,
    device: torch.device,
) -> tuple[VisionRelPosTransformer, Basis, dict[str, object], dict[str, object], dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = dict(checkpoint["args"])
    metadata = dict(checkpoint["metadata"])
    key = "best_model_state_dict" if state == "best" else "final_model_state_dict"
    state_dict = checkpoint[key]
    if state_dict is None:
        raise ValueError(f"checkpoint does not contain {key}")
    basis = rebuild_checkpoint_basis(checkpoint, teacher_bias=teacher_bias, device=device)
    model = VisionRelPosTransformer(
        basis,
        n_positions=int(metadata["grid_side"]) ** 2,
        patch_dim=int(state_dict["input.weight"].shape[1]) - 1,
        dim=int(ckpt_args["dim"]),
        n_heads=int(ckpt_args["n_heads"]),
        depth=int(ckpt_args["depth"]),
        ffn_mult=int(ckpt_args["ffn_mult"]),
        dropout=float(ckpt_args.get("dropout", 0.0)),
        n_classes=int(state_dict["classifier.weight"].shape[0]),
    ).to(device)
    model.load_state_dict({item_key: item_value.to(device) for item_key, item_value in state_dict.items()})
    model.eval()
    return model, basis, ckpt_args, metadata, dict(checkpoint.get("row", {}))


def column_mixed_ratio(matrix: torch.Tensor, *, side: int) -> torch.Tensor:
    n_positions = side * side
    cols = matrix.T.reshape(matrix.shape[1], n_positions, n_positions)
    tables = pairwise_to_relative_table(cols, side).to(torch.float64)
    centered = tables - tables.mean(dim=(-2, -1), keepdim=True)
    denom = torch.sqrt(torch.mean(centered.square(), dim=(-2, -1))).clamp_min(1e-12)
    mixed = tables[..., 1:, 1:] - tables[..., 1:, :-1] - tables[..., :-1, 1:] + tables[..., :-1, :-1]
    numer = torch.sqrt(torch.mean(mixed.square(), dim=(-2, -1)))
    return (numer / denom).to(torch.float32)


def sector_masks(
    basis: Basis,
    *,
    side: int,
    axial_mixed_threshold: float,
) -> dict[str, torch.Tensor]:
    n_features = basis.matrix.shape[1]
    labels = list(basis.labels)
    orders = list(basis.orders)
    device = basis.matrix.device
    idx = torch.arange(n_features, device=device)
    label_lower = [label.lower() for label in labels]
    is_const = torch.tensor([label == "const" or label.endswith(":const") for label in label_lower], device=device)
    order_tensor = torch.tensor([int(order) for order in orders], device=device)
    mixed_ratio = column_mixed_ratio(basis.matrix.to(torch.float32), side=side).to(device=device)

    label_axis = torch.tensor(
        [
            ("axis" in label)
            or ("dx_" in label)
            or ("dy_" in label)
            or ("axis0" in label)
            or ("axis1" in label)
            for label in label_lower
        ],
        device=device,
    )
    is_axial = (~is_const) & (label_axis | (mixed_ratio <= float(axial_mixed_threshold)))
    is_j0 = order_tensor == 0
    masks = {
        "const": is_const,
        "axial_J0": is_j0 & is_axial,
        "oblique_J0": is_j0 & (~is_axial) & (~is_const),
    }
    for order in sorted(set(int(item) for item in orders)):
        if order > 0:
            masks[f"jet_order_{order}"] = order_tensor == order
    assigned = torch.zeros(n_features, device=device, dtype=torch.bool)
    for mask in masks.values():
        assigned |= mask
    masks["residual_like"] = ~assigned
    masks = {name: mask for name, mask in masks.items() if bool(mask.any())}
    masks["all_positional"] = idx >= 0
    return masks


def coeff_mask_overrides(
    model: VisionRelPosTransformer,
    feature_mask: torch.Tensor,
    *,
    mode: str,
) -> list[torch.Tensor]:
    mask = feature_mask.to(device=model.basis_matrix.device, dtype=torch.bool)
    overrides: list[torch.Tensor] = []
    for block in model.blocks:
        coeff = block.coeff
        if mode == "keep":
            controlled = torch.zeros_like(coeff)
            controlled[:, mask] = coeff[:, mask]
        elif mode == "ablate":
            controlled = coeff.clone()
            controlled[:, mask] = 0.0
        else:
            raise ValueError(f"unknown coeff intervention mode: {mode}")
        bias = torch.einsum("nf,hf->hn", model.basis_matrix, controlled).reshape(
            block.n_heads,
            model.n_positions,
            model.n_positions,
        )
        overrides.append(bias)
    return overrides


def evaluate_metric(
    model: VisionRelPosTransformer,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    task: str,
    args: argparse.Namespace,
    seed: int,
    bias_overrides: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    if task == "classification":
        return evaluate_classifier(model, x, y, args=args, bias_overrides=bias_overrides)
    if task == "reconstruction":
        return evaluate_reconstruction(model, x, args=args, seed=seed, bias_overrides=bias_overrides)
    raise ValueError(f"unknown task: {task}")


def add_eval_row(
    rows: list[dict[str, object]],
    *,
    checkpoint_path: Path,
    metadata: dict[str, object],
    row_metadata: dict[str, object],
    state: str,
    mode: str,
    sector: str,
    n_features: int,
    metric: dict[str, float],
    full_score: float,
    zero_score: float,
    coeff_energy_share: float,
) -> None:
    score = float(metric["score"])
    rows.append(
        {
            "checkpoint": str(checkpoint_path),
            "state": state,
            "dataset": metadata.get("dataset"),
            "task": metadata.get("task"),
            "basis": metadata.get("basis"),
            "seed": metadata.get("seed"),
            "train_steps": metadata.get("steps"),
            "checkpoint_score": row_metadata.get("score", metadata.get("score")),
            "checkpoint_final_score": row_metadata.get("final_score", metadata.get("final_score")),
            "eval_mode": mode,
            "sector": sector,
            "sector_features": int(n_features),
            "score": score,
            "loss": float(metric["loss"]),
            "delta_vs_full": score - full_score,
            "gain_vs_zero": score - zero_score,
            "coeff_energy_share": coeff_energy_share,
        }
    )


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["task"]), str(row["eval_mode"]), str(row["sector"]))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, object]] = []
    for (dataset, task, mode, sector), values in sorted(groups.items()):
        scores = np.array([float(row["score"]) for row in values], dtype=np.float64)
        deltas = np.array([float(row["delta_vs_full"]) for row in values], dtype=np.float64)
        gains = np.array([float(row["gain_vs_zero"]) for row in values], dtype=np.float64)
        energy = np.array([float(row["coeff_energy_share"]) for row in values], dtype=np.float64)
        out.append(
            {
                "dataset": dataset,
                "task": task,
                "eval_mode": mode,
                "sector": sector,
                "n": len(values),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std(ddof=0)),
                "delta_vs_full_mean": float(deltas.mean()),
                "gain_vs_zero_mean": float(gains.mean()),
                "coeff_energy_share_mean": float(energy.mean()),
                "sector_features": int(values[0]["sector_features"]),
            }
        )
    return out


def write_report(output_dir: Path, summary: dict[str, object], aggregate: list[dict[str, object]]) -> None:
    lines = [
        "# V12 Sector Intervention Report",
        "",
        "Environment:",
        "",
        f"- Device: {summary['device']}",
        f"- State: {summary['state']}",
        f"- Dataset: {summary['dataset']}",
        f"- Task: {summary['task']}",
        f"- Checkpoints: {summary['num_checkpoints']}",
        f"- Rows: {summary['rows']}",
        "",
        "## Aggregate",
        "",
        "| mode | sector | n | features | score mean | score std | delta vs full | gain vs zero | coeff energy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| "
            + f"{row['eval_mode']} | {row['sector']} | {int(row['n'])} | {int(row['sector_features'])} | "
            + f"{float(row['score_mean']):.4f} | {float(row['score_std']):.4f} | "
            + f"{float(row['delta_vs_full_mean']):.4f} | {float(row['gain_vs_zero_mean']):.4f} | "
            + f"{float(row['coeff_energy_share_mean']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `keep` evaluates only the selected positional-bias sector.",
            "- `ablate` evaluates the full positional bias with that sector removed.",
            "- Coeff energy share is measured on learned positional coefficients, not on task loss.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    cwd = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = [resolve_path(item, cwd=cwd) for item in parse_list(args.checkpoints)]
    if not checkpoints:
        raise ValueError("--checkpoints must name at least one checkpoint")
    teacher_bias = resolve_path(args.teacher_bias, cwd=cwd) if args.teacher_bias else None

    all_rows: list[dict[str, object]] = []
    sector_rows: list[dict[str, object]] = []
    for checkpoint_path in checkpoints:
        model, basis, ckpt_args, metadata, row_metadata = load_checkpoint_model(
            checkpoint_path,
            state=args.state,
            teacher_bias=teacher_bias,
            device=device,
        )
        data_root = Path(args.data_root or ckpt_args.get("data_root", "data"))
        dataset = args.dataset or str(ckpt_args.get("dataset", metadata.get("dataset", "mnist")))
        task = args.task or str(ckpt_args.get("task", metadata.get("task", "classification")))
        train_limit = args.train_limit if args.train_limit is not None else ckpt_args.get("train_limit")
        test_limit = args.test_limit if args.test_limit is not None else ckpt_args.get("test_limit")
        train_images, train_y, test_images, test_y = load_vision_dataset(
            dataset=dataset,
            root=data_root,
            device=device,
            train_limit=train_limit,
            test_limit=test_limit,
        )
        train_x, grid_side, _ = patchify(train_images, int(ckpt_args["patch_size"]))
        test_x, test_grid_side, _ = patchify(test_images, int(ckpt_args["patch_size"]))
        if grid_side != test_grid_side or grid_side != int(metadata["grid_side"]):
            raise ValueError(f"checkpoint/data grid side mismatch for {checkpoint_path}")
        _, test_x = normalize_patches(train_x, test_x)
        eval_args = argparse.Namespace(
            eval_batch_size=args.eval_batch_size or int(ckpt_args.get("eval_batch_size", 2048)),
            amp=args.amp or ckpt_args.get("amp", "bf16"),
            mask_rate=args.mask_rate if args.mask_rate is not None else float(ckpt_args.get("mask_rate", 0.35)),
        )
        eval_seed = args.eval_seed if args.eval_seed is not None else int(metadata.get("seed", ckpt_args.get("seed", 426))) + 3000

        full_metric = evaluate_metric(model, test_x, test_y, task=task, args=eval_args, seed=eval_seed)
        zero_metric = evaluate_metric(
            model,
            test_x,
            test_y,
            task=task,
            args=eval_args,
            seed=eval_seed,
            bias_overrides=zero_bias_overrides(model),
        )
        full_score = float(full_metric["score"])
        zero_score = float(zero_metric["score"])
        add_eval_row(
            all_rows,
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            row_metadata=row_metadata,
            state=args.state,
            mode="full",
            sector="all_positional",
            n_features=model.basis_matrix.shape[1],
            metric=full_metric,
            full_score=full_score,
            zero_score=zero_score,
            coeff_energy_share=1.0,
        )
        add_eval_row(
            all_rows,
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            row_metadata=row_metadata,
            state=args.state,
            mode="zero_bias",
            sector="none",
            n_features=0,
            metric=zero_metric,
            full_score=full_score,
            zero_score=zero_score,
            coeff_energy_share=0.0,
        )

        masks = sector_masks(basis, side=grid_side, axial_mixed_threshold=args.axial_mixed_threshold)
        coeff = torch.stack([block.coeff.detach().float() for block in model.blocks], dim=0)
        total_energy = float(coeff.square().sum().detach().cpu())
        for sector_name in parse_list(args.sectors):
            if sector_name not in masks:
                continue
            mask = masks[sector_name].to(device=device)
            sector_energy = float(coeff[..., mask].square().sum().detach().cpu())
            sector_rows.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "basis": metadata.get("basis"),
                    "seed": metadata.get("seed"),
                    "sector": sector_name,
                    "sector_features": int(mask.sum().detach().cpu()),
                    "coeff_energy": sector_energy,
                    "coeff_energy_share": sector_energy / max(total_energy, 1e-12),
                }
            )
            for mode in parse_list(args.modes):
                metric = evaluate_metric(
                    model,
                    test_x,
                    test_y,
                    task=task,
                    args=eval_args,
                    seed=eval_seed,
                    bias_overrides=coeff_mask_overrides(model, mask, mode=mode),
                )
                add_eval_row(
                    all_rows,
                    checkpoint_path=checkpoint_path,
                    metadata=metadata,
                    row_metadata=row_metadata,
                    state=args.state,
                    mode=mode,
                    sector=sector_name,
                    n_features=int(mask.sum().detach().cpu()),
                    metric=metric,
                    full_score=full_score,
                    zero_score=zero_score,
                    coeff_energy_share=sector_energy / max(total_energy, 1e-12),
                )

    aggregate = aggregate_rows(all_rows)
    write_csv(output_dir / "sector_intervention_results.csv", all_rows)
    write_csv(output_dir / "sector_intervention_aggregate.csv", aggregate)
    write_csv(output_dir / "sector_energy.csv", sector_rows)
    summary = {
        "device": str(device),
        "state": args.state,
        "dataset": args.dataset or "from_checkpoint",
        "task": args.task or "from_checkpoint",
        "num_checkpoints": len(checkpoints),
        "rows": len(all_rows),
        "teacher_bias": str(teacher_bias) if teacher_bias is not None else "",
        "output_dir": str(output_dir),
        "results_csv": str(output_dir / "sector_intervention_results.csv"),
        "aggregate_csv": str(output_dir / "sector_intervention_aggregate.csv"),
    }
    write_json(output_dir / "sector_intervention_summary.json", summary)
    write_report(output_dir, summary, aggregate)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sector keep/ablate interventions from positional-bias checkpoints.")
    parser.add_argument("--checkpoints", required=True, help="Comma-separated checkpoint .pt files.")
    parser.add_argument("--teacher-bias", type=str, default="", help="Teacher bias NPZ used to rebuild basis labels.")
    parser.add_argument("--state", choices=["final", "best"], default="final")
    parser.add_argument("--sectors", type=str, default="const,axial_J0,oblique_J0,jet_order_1,jet_order_2,residual_like")
    parser.add_argument("--modes", type=str, default="keep,ablate")
    parser.add_argument("--axial-mixed-threshold", type=float, default=1e-5)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--task", choices=["classification", "reconstruction"], default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--amp", choices=["none", "bf16", "fp16"], default=None)
    parser.add_argument("--mask-rate", type=float, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
