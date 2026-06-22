from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from toric_pj.diagnostics.basis_projection import Basis, default_device
from toric_pj.diagnostics.relative_table_geometry import load_bias_npz, pairwise_to_relative_table
from toric_pj.experiments.v3_real_vision_scaling import (
    TinyImageNetDataset,
    VisionRelPosTransformer,
    evaluate_reconstruction,
    patchify,
    vision_num_classes,
)
from toric_pj.experiments.v4_metric_toric_pj import build_teacher_bases, parse_list
from toric_pj.experiments.real_digits_probe import make_positions, pairwise_d, relative_2d_table_basis
from toric_pj.experiments.v4_offset_holdout import zero_bias_overrides


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
    candidates = [path, cwd / path, cwd / "MetricToric" / path]
    if path.parts and path.parts[0] == "results":
        candidates.append(cwd / "MetricToric" / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def checkpoint_paths(values: str, *, cwd: Path) -> list[Path]:
    out: list[Path] = []
    for item in parse_list(values):
        path = resolve_path(item, cwd=cwd)
        if path.is_dir():
            ckpt_dir = path / "checkpoints" if (path / "checkpoints").is_dir() else path
            out.extend(sorted(ckpt_dir.glob("*.pt")))
        else:
            out.append(path)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in out:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def load_images(
    *,
    dataset: str,
    root: Path,
    split: str,
    resolution: int,
    device: torch.device,
    limit: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    transform = transforms.Compose([transforms.Resize((resolution, resolution)), transforms.ToTensor()])
    name = dataset.lower()
    if name == "tiny-imagenet":
        ds = TinyImageNetDataset(root, split="train" if split == "train" else "val", transform=transform)
    elif name == "stl10":
        ds = datasets.STL10(root=str(root), split="train" if split == "train" else "test", download=False, transform=transform)
    elif name == "cifar10":
        ds = datasets.CIFAR10(root=str(root), train=(split == "train"), download=False, transform=transform)
    elif name == "cifar100":
        ds = datasets.CIFAR100(root=str(root), train=(split == "train"), download=False, transform=transform)
    else:
        raise ValueError(f"unsupported E6 dataset: {dataset}")

    count = len(ds) if limit is None else min(int(limit), len(ds))
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    for idx in range(count):
        x, y = ds[idx]
        xs.append(x)
        ys.append(int(y))
    return torch.stack(xs, dim=0).to(device=device, dtype=torch.float32), torch.tensor(ys, device=device, dtype=torch.long)


def patch_stats(train_images: torch.Tensor, *, patch_size: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    train_x, train_side, patch_dim = patchify(train_images, patch_size)
    mean = train_x.mean(dim=(0, 1), keepdim=True)
    std = train_x.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    return mean, std, train_side, patch_dim


def pairwise_offsets(side: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    positions = make_positions(side, device).to(dtype=dtype)
    return pairwise_d(positions).reshape(-1, 2).to(dtype=dtype)


def rel_offset_index(side: int, labels: list[str]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for idx, label in enumerate(labels):
        if not label.startswith("rel_"):
            continue
        _, sx, sy = label.split("_", 2)
        out[(int(sx), int(sy))] = idx
    if len(out) != len(labels):
        raise ValueError("relative table labels are not complete rel_dx_dy labels")
    return out


def relative_extension_matrix(
    *,
    checkpoint_basis_matrix: torch.Tensor,
    train_side: int,
    eval_side: int,
    mode: str,
    device: torch.device,
) -> Basis:
    train_basis = relative_2d_table_basis(pairwise_offsets(train_side, device=device, dtype=torch.float64))
    labels = list(train_basis.labels)
    orders = list(train_basis.orders)
    offset_to_idx = rel_offset_index(train_side, labels)
    col_values = checkpoint_basis_matrix.to(device=device, dtype=torch.float32).abs().amax(dim=0).clamp_min(1e-12)
    d_eval = pairwise_offsets(eval_side, device=device, dtype=torch.float32)
    radius = train_side - 1
    matrix = torch.zeros((d_eval.shape[0], len(labels)), device=device, dtype=torch.float32)

    if mode == "table_clamp":
        clipped = d_eval.round().clamp(-radius, radius).to(torch.long)
        for row_idx, (dx, dy) in enumerate(clipped.detach().cpu().tolist()):
            col = offset_to_idx[(int(dx), int(dy))]
            matrix[row_idx, col] = col_values[col]
    elif mode == "scaled_interp":
        scale = float(train_side - 1) / float(max(eval_side - 1, 1))
        coords = (d_eval * scale).clamp(-float(radius), float(radius))
        x0 = torch.floor(coords[:, 0]).to(torch.long).clamp(-radius, radius)
        y0 = torch.floor(coords[:, 1]).to(torch.long).clamp(-radius, radius)
        x1 = (x0 + 1).clamp(-radius, radius)
        y1 = (y0 + 1).clamp(-radius, radius)
        wx = (coords[:, 0] - x0.to(coords.dtype)).clamp(0.0, 1.0)
        wy = (coords[:, 1] - y0.to(coords.dtype)).clamp(0.0, 1.0)
        corners = [
            (x0, y0, (1.0 - wx) * (1.0 - wy)),
            (x1, y0, wx * (1.0 - wy)),
            (x0, y1, (1.0 - wx) * wy),
            (x1, y1, wx * wy),
        ]
        for xs, ys, weights in corners:
            for row_idx, (dx, dy, weight) in enumerate(zip(xs.detach().cpu().tolist(), ys.detach().cpu().tolist(), weights.detach().cpu().tolist())):
                if weight == 0.0:
                    continue
                col = offset_to_idx[(int(dx), int(dy))]
                matrix[row_idx, col] += float(weight) * col_values[col]
    else:
        raise ValueError(f"unknown relative extension mode: {mode}")

    return Basis(f"relative_2d_table_{mode}_from{train_side}_to{eval_side}", matrix, labels, orders)


def load_teacher_tables(checkpoint: dict[str, object], *, cwd: Path, device: torch.device) -> torch.Tensor:
    args = checkpoint.get("args", {})
    if not isinstance(args, dict) or not args.get("teacher_bias"):
        raise ValueError("checkpoint does not record teacher_bias")
    teacher_path = resolve_path(str(args["teacher_bias"]), cwd=cwd)
    bundle, _ = load_bias_npz(teacher_path, device=torch.device("cpu"))
    return bundle.tables.float().to(device=device)


def checkpoint_seed(checkpoint: dict[str, object]) -> int:
    metadata = checkpoint.get("metadata", {})
    if isinstance(metadata, dict) and "seed" in metadata:
        return int(metadata["seed"])
    args = checkpoint.get("args", {})
    if isinstance(args, dict) and "seed" in args:
        return int(args["seed"])
    return 426


def functional_extension_basis(
    checkpoint: dict[str, object],
    *,
    train_side: int,
    eval_side: int,
    cwd: Path,
    device: torch.device,
) -> Basis:
    basis_name = str(checkpoint.get("basis_name", checkpoint.get("metadata", {}).get("basis", "")))
    checkpoint_basis = checkpoint["basis_matrix"].to(device=device, dtype=torch.float32)
    if basis_name == "relative_2d_table":
        return relative_extension_matrix(
            checkpoint_basis_matrix=checkpoint_basis,
            train_side=train_side,
            eval_side=eval_side,
            mode="scaled_interp",
            device=device,
        )

    teacher_tables = load_teacher_tables(checkpoint, cwd=cwd, device=device)
    seed = checkpoint_seed(checkpoint)
    train_basis = build_teacher_bases(
        side=train_side,
        device=device,
        teacher_tables=teacher_tables,
        variants=[basis_name],
        include_shuffle=False,
        seed=seed,
    )[0]
    eval_basis = build_teacher_bases(
        side=eval_side,
        device=device,
        teacher_tables=teacher_tables,
        variants=[basis_name],
        include_shuffle=False,
        seed=seed,
    )[0]
    if train_basis.matrix.shape[1] != checkpoint_basis.shape[1] or eval_basis.matrix.shape[1] != checkpoint_basis.shape[1]:
        raise ValueError(
            f"basis feature mismatch for {basis_name}: "
            f"checkpoint={checkpoint_basis.shape[1]}, train={train_basis.matrix.shape[1]}, eval={eval_basis.matrix.shape[1]}"
        )
    train_norms = torch.linalg.norm(train_basis.matrix.to(torch.float32), dim=0).clamp_min(1e-12)
    matrix = eval_basis.matrix.to(device=device, dtype=torch.float32) / train_norms.unsqueeze(0)
    return Basis(f"{basis_name}_functional_from{train_side}_to{eval_side}", matrix, list(eval_basis.labels), list(eval_basis.orders))


def instantiate_eval_model(
    checkpoint: dict[str, object],
    *,
    state: str,
    basis: Basis,
    eval_side: int,
    patch_dim: int,
    device: torch.device,
) -> VisionRelPosTransformer:
    ckpt_args = dict(checkpoint["args"])
    metadata = dict(checkpoint["metadata"])
    key = "best_model_state_dict" if state == "best" else "final_model_state_dict"
    state_dict = checkpoint[key]
    if state_dict is None:
        raise ValueError(f"checkpoint does not contain {key}")
    model = VisionRelPosTransformer(
        basis,
        n_positions=eval_side * eval_side,
        patch_dim=patch_dim,
        dim=int(ckpt_args["dim"]),
        n_heads=int(ckpt_args["n_heads"]),
        depth=int(ckpt_args["depth"]),
        ffn_mult=int(ckpt_args["ffn_mult"]),
        dropout=float(ckpt_args.get("dropout", 0.0)),
        n_classes=vision_num_classes(str(metadata["dataset"])),
    ).to(device)
    filtered = {k: v.to(device) for k, v in state_dict.items() if k != "basis_matrix"}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    unexpected_clean = [item for item in unexpected if item != "basis_matrix"]
    missing_clean = [item for item in missing if item != "basis_matrix"]
    if missing_clean or unexpected_clean:
        raise ValueError(f"state dict mismatch: missing={missing_clean}, unexpected={unexpected_clean}")
    model.basis_matrix.copy_(basis.matrix.to(device=device, dtype=torch.float32))
    model.eval()
    return model


def bias_overrides_from_eval_model(model: VisionRelPosTransformer) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for block in model.blocks:
        bias = torch.einsum("nf,hf->hn", model.basis_matrix, block.coeff).reshape(
            block.n_heads,
            model.n_positions,
            model.n_positions,
        )
        out.append(bias)
    return out


def extend_train_tables(
    tables: torch.Tensor,
    *,
    train_side: int,
    eval_side: int,
    mode: str,
) -> torch.Tensor:
    device = tables.device
    dtype = tables.dtype
    d_eval = pairwise_offsets(eval_side, device=device, dtype=torch.float32)
    radius = train_side - 1
    shift = radius
    flat = torch.zeros((*tables.shape[:2], d_eval.shape[0]), device=device, dtype=dtype)

    if mode == "table_clamp":
        clipped = d_eval.round().clamp(-radius, radius).to(torch.long)
        ix = clipped[:, 0] + shift
        iy = clipped[:, 1] + shift
        flat = tables[..., ix, iy]
    elif mode == "scaled_interp":
        scale = float(train_side - 1) / float(max(eval_side - 1, 1))
        coords = (d_eval * scale).clamp(-float(radius), float(radius))
        x0 = torch.floor(coords[:, 0]).to(torch.long).clamp(-radius, radius)
        y0 = torch.floor(coords[:, 1]).to(torch.long).clamp(-radius, radius)
        x1 = (x0 + 1).clamp(-radius, radius)
        y1 = (y0 + 1).clamp(-radius, radius)
        wx = (coords[:, 0] - x0.to(coords.dtype)).clamp(0.0, 1.0).to(dtype)
        wy = (coords[:, 1] - y0.to(coords.dtype)).clamp(0.0, 1.0).to(dtype)
        flat = (
            tables[..., x0 + shift, y0 + shift] * ((1.0 - wx) * (1.0 - wy))
            + tables[..., x1 + shift, y0 + shift] * (wx * (1.0 - wy))
            + tables[..., x0 + shift, y1 + shift] * ((1.0 - wx) * wy)
            + tables[..., x1 + shift, y1 + shift] * (wx * wy)
        )
    else:
        raise ValueError(f"unknown table extension mode: {mode}")
    return flat.reshape(*tables.shape[:2], eval_side * eval_side, eval_side * eval_side)


def train_tables_from_checkpoint_model(model: VisionRelPosTransformer, *, train_side: int) -> torch.Tensor:
    full = torch.stack(bias_overrides_from_eval_model(model), dim=0).float()
    return pairwise_to_relative_table(full, train_side)


def radial_truncate_overrides(overrides: list[torch.Tensor], *, eval_side: int, radius: float) -> list[torch.Tensor]:
    vals = torch.arange(eval_side, device=overrides[0].device, dtype=torch.float32)
    pos = torch.stack(torch.meshgrid(vals, vals, indexing="ij"), dim=-1).reshape(-1, 2)
    d = pos[:, None, :] - pos[None, :, :]
    keep = torch.linalg.norm(d, dim=-1) <= float(radius)
    mask = keep.to(device=overrides[0].device, dtype=overrides[0].dtype)
    return [bias * mask.unsqueeze(0) for bias in overrides]


def bias_stats(overrides: list[torch.Tensor], *, eval_side: int, train_side: int, holdout_radius: float) -> dict[str, float]:
    full = torch.stack(overrides, dim=0).float()
    tables = pairwise_to_relative_table(full, eval_side)
    vals = torch.arange(-(eval_side - 1), eval_side, device=tables.device, dtype=torch.float32)
    gx, gy = torch.meshgrid(vals, vals, indexing="ij")
    radius = torch.sqrt(gx.square() + gy.square())

    def rms(mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return float("nan")
        return float(torch.sqrt(torch.mean(tables[..., mask].square())).detach().cpu())

    def maxabs(mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return float("nan")
        return float(torch.max(torch.abs(tables[..., mask])).detach().cpu())

    all_mask = torch.ones_like(radius, dtype=torch.bool)
    near_mask = radius <= float(holdout_radius)
    far_train_mask = radius > float(train_side - 1)
    return {
        "bias_rms_all": rms(all_mask),
        "bias_max_all": maxabs(all_mask),
        "bias_rms_near_r": rms(near_mask),
        "bias_max_near_r": maxabs(near_mask),
        "bias_rms_beyond_train_radius": rms(far_train_mask),
        "bias_max_beyond_train_radius": maxabs(far_train_mask),
    }


def method_label(checkpoint_path: Path, checkpoint: dict[str, object]) -> str:
    basis = str(checkpoint.get("basis_name", checkpoint.get("metadata", {}).get("basis", "")))
    if basis == "relative_2d_table":
        return "relative_table"
    if basis.startswith("dct_top"):
        return basis
    if basis.startswith("table_informed_toric_PJ"):
        return "toric_pj"
    if basis.startswith("mixed_toric_PJ") and "farband_teacher" in str(checkpoint_path):
        return "farband_repair"
    if basis.startswith("mixed_toric_PJ"):
        return "mixed_toric_dct"
    return basis


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["method"]),
            str(row["basis"]),
            str(row["extension"]),
            int(row["eval_resolution"]),
            int(row["eval_side"]),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, object]] = []
    for (method, basis, extension, resolution, side), items in sorted(groups.items()):
        scores = np.array([float(item["score"]) for item in items], dtype=np.float64)
        losses = np.array([float(item["loss"]) for item in items], dtype=np.float64)
        row: dict[str, object] = {
            "method": method,
            "basis": basis,
            "extension": extension,
            "eval_resolution": resolution,
            "eval_side": side,
            "n": len(items),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "loss_mean": float(losses.mean()),
        }
        for key in [
            "bias_rms_all",
            "bias_max_all",
            "bias_rms_near_r",
            "bias_max_near_r",
            "bias_rms_beyond_train_radius",
            "bias_max_beyond_train_radius",
        ]:
            vals = np.array([float(item[key]) for item in items], dtype=np.float64)
            row[f"{key}_mean"] = float(np.nanmean(vals))
        out.append(row)
    return out


def write_report(output_dir: Path, summary: dict[str, object], aggregate: list[dict[str, object]]) -> None:
    lines = [
        "# V12 E6 Resolution Transfer Report",
        "",
        "This is a fixed-weight evaluation. Transformer weights and positional coefficients are loaded from existing checkpoints; only the input resize/grid and positional-bias extension rule change.",
        "",
        "## Summary",
        "",
        f"- Dataset: {summary['dataset']}",
        f"- Train resolution / patch size / grid: {summary['train_resolution']} / {summary['patch_size']} / {summary['train_side']}",
        f"- Eval resolutions: {summary['eval_resolutions']}",
        f"- State: {summary['state']}",
        f"- Test limit: {summary['test_limit']}",
        "",
        "## Aggregate",
        "",
        "| method | extension | resolution | grid | n | score mean | score std | far bias RMS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            "| "
            + f"{row['method']} | {row['extension']} | {int(row['eval_resolution'])} | {int(row['eval_side'])} | "
            + f"{int(row['n'])} | {float(row['score_mean']):.4f} | {float(row['score_std']):.4f} | "
            + f"{float(row['bias_rms_beyond_train_radius_mean']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Resize distribution shift is shared by all methods and controls; do not attribute all score changes to positional encoding.",
            "- `functional` is the native compact-basis extension, while `table_clamp` extends each checkpoint's learned train-window bias table by nearest boundary clamping.",
            "- For `relative_2d_table`, `functional` is scaled bilinear interpolation from the train-window table.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    cwd = Path.cwd()
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    checkpoints = checkpoint_paths(args.train_checkpoints, cwd=cwd)
    if not checkpoints:
        raise ValueError("no checkpoints found")

    norm_train_images, _ = load_images(
        dataset=args.dataset,
        root=Path(args.data_root),
        split="train",
        resolution=args.train_resolution,
        device=device,
        limit=args.norm_train_limit,
    )
    mean, std, train_side, patch_dim = patch_stats(norm_train_images, patch_size=args.patch_size)
    if train_side * args.patch_size != args.train_resolution:
        raise ValueError("train resolution and patch size do not match train grid")
    del norm_train_images

    eval_resolutions = parse_int_list(args.eval_resolutions)
    controls = set(parse_list(args.controls))
    rows: list[dict[str, object]] = []
    eval_cache: dict[int, torch.Tensor] = {}

    eval_args = argparse.Namespace(eval_batch_size=args.eval_batch_size, mask_rate=args.mask_rate, amp=args.amp)
    for resolution in eval_resolutions:
        test_images, test_y = load_images(
            dataset=args.dataset,
            root=Path(args.data_root),
            split="test",
            resolution=resolution,
            device=device,
            limit=args.test_limit,
        )
        test_x, eval_side, eval_patch_dim = patchify(test_images, args.patch_size)
        if eval_patch_dim != patch_dim:
            raise ValueError(f"patch dim mismatch at resolution {resolution}: {eval_patch_dim} != {patch_dim}")
        eval_cache[resolution] = (test_x - mean) / std
        del test_images, test_y

    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        metadata = dict(checkpoint["metadata"])
        ckpt_args = dict(checkpoint["args"])
        if int(metadata.get("grid_side", train_side)) != train_side:
            raise ValueError(f"checkpoint grid {metadata.get('grid_side')} != train side {train_side}: {checkpoint_path}")
        if int(ckpt_args.get("patch_size", args.patch_size)) != args.patch_size:
            raise ValueError(f"checkpoint patch size {ckpt_args.get('patch_size')} != {args.patch_size}: {checkpoint_path}")
        seed = checkpoint_seed(checkpoint)
        basis_name = str(checkpoint.get("basis_name", metadata.get("basis", "")))
        method = method_label(checkpoint_path, checkpoint)

        train_basis = functional_extension_basis(
            checkpoint,
            train_side=train_side,
            eval_side=train_side,
            cwd=cwd,
            device=device,
        )
        train_model = instantiate_eval_model(
            checkpoint,
            state=args.state,
            basis=train_basis,
            eval_side=train_side,
            patch_dim=patch_dim,
            device=device,
        )
        train_tables = train_tables_from_checkpoint_model(train_model, train_side=train_side)
        del train_model

        for resolution in eval_resolutions:
            test_x = eval_cache[resolution]
            eval_side = int(test_x.shape[1] ** 0.5)
            basis = functional_extension_basis(
                checkpoint,
                train_side=train_side,
                eval_side=eval_side,
                cwd=cwd,
                device=device,
            )
            model = instantiate_eval_model(
                checkpoint,
                state=args.state,
                basis=basis,
                eval_side=eval_side,
                patch_dim=patch_dim,
                device=device,
            )
            base_overrides = bias_overrides_from_eval_model(model)
            modes: list[tuple[str, list[torch.Tensor] | None]] = [("functional", None)]
            if "no_pos" in controls:
                modes.append(("no_pos", zero_bias_overrides(model)))
            if "table_clamp" in controls:
                pairwise = extend_train_tables(train_tables, train_side=train_side, eval_side=eval_side, mode="table_clamp")
                modes.append(("table_clamp", [pairwise[layer] for layer in range(pairwise.shape[0])]))
            if "table_scaled_interp" in controls:
                pairwise = extend_train_tables(train_tables, train_side=train_side, eval_side=eval_side, mode="scaled_interp")
                modes.append(("table_scaled_interp", [pairwise[layer] for layer in range(pairwise.shape[0])]))
            for control in controls:
                if control.startswith("radial_truncate_r"):
                    radius = float(control.rsplit("r", 1)[1].replace("p", "."))
                    modes.append((control, radial_truncate_overrides(base_overrides, eval_side=eval_side, radius=radius)))

            for extension, overrides in modes:
                active_overrides = base_overrides if overrides is None else overrides
                metric = evaluate_reconstruction(
                    model,
                    test_x,
                    args=eval_args,
                    seed=seed + 10_000 + eval_side,
                    bias_overrides=overrides,
                )
                stats = bias_stats(
                    active_overrides,
                    eval_side=eval_side,
                    train_side=train_side,
                    holdout_radius=float(args.holdout_radius),
                )
                rows.append(
                    {
                        "checkpoint": str(checkpoint_path),
                        "method": method,
                        "basis": basis_name,
                        "seed": seed,
                        "state": args.state,
                        "extension": extension,
                        "dataset": args.dataset,
                        "train_resolution": args.train_resolution,
                        "eval_resolution": resolution,
                        "train_side": train_side,
                        "eval_side": eval_side,
                        "patch_size": args.patch_size,
                        "test_count": int(test_x.shape[0]),
                        "score": float(metric["score"]),
                        "loss": float(metric["loss"]),
                        **stats,
                    }
                )
            del model
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()

    aggregate = aggregate_rows(rows)
    write_csv(output_dir / "resolution_transfer_results.csv", rows)
    write_csv(output_dir / "resolution_transfer_aggregate.csv", aggregate)
    peak_mem = int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() and device.type == "cuda" else 0
    summary = {
        "device": str(device),
        "dataset": args.dataset,
        "train_resolution": args.train_resolution,
        "eval_resolutions": eval_resolutions,
        "patch_size": args.patch_size,
        "train_side": train_side,
        "patch_dim": patch_dim,
        "state": args.state,
        "checkpoints": [str(path) for path in checkpoints],
        "controls": sorted(controls),
        "norm_train_limit": args.norm_train_limit,
        "test_limit": args.test_limit,
        "rows": len(rows),
        "peak_cuda_memory_bytes": peak_mem,
    }
    write_json(output_dir / "resolution_transfer_summary.json", summary)
    write_report(output_dir, summary, aggregate)
    summary["summary"] = str(output_dir / "resolution_transfer_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V12 E6 fixed-weight resolution transfer evaluation.")
    parser.add_argument("--train-checkpoints", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="tiny-imagenet")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--train-resolution", type=int, default=32)
    parser.add_argument("--eval-resolutions", type=str, default="32,40,48,64")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--controls", type=str, default="no_pos,table_clamp,radial_truncate_r4")
    parser.add_argument("--holdout-radius", type=float, default=4.0)
    parser.add_argument("--state", choices=["final", "best"], default="best")
    parser.add_argument("--norm-train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--mask-rate", type=float, default=0.35)
    parser.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="MetricToric/results/v12_e6_resolution_transfer_tinyimagenet")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
