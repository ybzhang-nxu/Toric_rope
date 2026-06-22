from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from toric_pj.diagnostics.basis_projection import Basis, default_device
from toric_pj.diagnostics.relative_table_geometry import pairwise_to_relative_table, relative_table_to_pairwise
from toric_pj.experiments.v3_real_vision_scaling import (
    VisionRelPosTransformer,
    evaluate_classifier,
    evaluate_reconstruction,
    load_vision_dataset,
    normalize_patches,
    patchify,
)
from toric_pj.experiments.v4_offset_holdout import (
    bias_overrides_from_model,
    component_radial_bias_overrides,
    eval_control_bias_overrides,
    parse_float_list,
    zero_bias_overrides,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def parse_head_subset_spec(value: str) -> list[tuple[str, list[tuple[int, list[int]]]]]:
    out: list[tuple[str, list[tuple[int, list[int]]]]] = []
    for raw_subset in value.split(";"):
        raw_subset = raw_subset.strip()
        if not raw_subset:
            continue
        components: list[tuple[int, list[int]]] = []
        labels: list[str] = []
        for raw_component in raw_subset.split(","):
            raw_component = raw_component.strip()
            if not raw_component:
                continue
            if ":" not in raw_component:
                raise ValueError(f"head subset component must be layer:head+head, got {raw_component!r}")
            raw_layer, raw_heads = raw_component.split(":", 1)
            layer = int(raw_layer.strip().removeprefix("L").removeprefix("l"))
            heads = [
                int(item.strip().removeprefix("H").removeprefix("h"))
                for item in raw_heads.replace("|", "+").split("+")
                if item.strip()
            ]
            if not heads:
                raise ValueError(f"head subset component has no heads: {raw_component!r}")
            components.append((layer, heads))
            labels.append(f"L{layer}" + "+".join(f"H{head}" for head in heads))
        out.append(("+".join(labels), components))
    return out


def subset_radial_bias_overrides(
    model: VisionRelPosTransformer,
    *,
    components: list[tuple[int, list[int]]],
    radius: float,
    mode: str,
) -> list[torch.Tensor]:
    side = int(math.sqrt(float(model.n_positions)))
    full = torch.stack(bias_overrides_from_model(model, pairwise_visible=None), dim=0).float()
    tables = pairwise_to_relative_table(full, side)
    vals = torch.arange(-(side - 1), side, device=tables.device, dtype=torch.float32)
    gx, gy = torch.meshgrid(vals, vals, indexing="ij")
    mask = (torch.sqrt(gx.square() + gy.square()) <= float(radius)).to(device=tables.device, dtype=tables.dtype)
    n_layers, n_heads = tables.shape[:2]
    if mode == "head_subset_keep":
        controlled = torch.zeros_like(tables)
        for layer, heads in components:
            if layer < 0 or layer >= n_layers:
                raise ValueError(f"layer index out of range: {layer}")
            for head in heads:
                if head < 0 or head >= n_heads:
                    raise ValueError(f"head index out of range: {head}")
                controlled[layer, head] = tables[layer, head] * mask
    elif mode == "head_subset_ablate":
        controlled = tables.clone()
        for layer, heads in components:
            if layer < 0 or layer >= n_layers:
                raise ValueError(f"layer index out of range: {layer}")
            for head in heads:
                if head < 0 or head >= n_heads:
                    raise ValueError(f"head index out of range: {head}")
                controlled[layer, head] = controlled[layer, head] * (1.0 - mask)
    else:
        raise ValueError(f"unknown subset mode: {mode}")
    pairwise = relative_table_to_pairwise(controlled, side)
    return [pairwise[layer] for layer in range(pairwise.shape[0])]


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    state: str,
    device: torch.device,
) -> tuple[VisionRelPosTransformer, dict[str, object], dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = dict(checkpoint["args"])
    key = "best_model_state_dict" if state == "best" else "final_model_state_dict"
    state_dict = checkpoint[key]
    if state_dict is None:
        raise ValueError(f"checkpoint does not contain {key}")
    basis_matrix = checkpoint["basis_matrix"].to(device=device, dtype=torch.float32)
    basis = Basis(
        str(checkpoint.get("basis_name", checkpoint["metadata"].get("basis", "checkpoint_basis"))),
        basis_matrix,
        [f"f{i}" for i in range(basis_matrix.shape[1])],
        list(checkpoint.get("basis_orders", [0] * basis_matrix.shape[1])),
    )
    model = VisionRelPosTransformer(
        basis,
        n_positions=int(checkpoint["metadata"]["grid_side"]) ** 2,
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
    return model, ckpt_args, dict(checkpoint["metadata"])


def add_metric_row(
    rows: list[dict[str, object]],
    *,
    mode: str,
    param: str,
    metric: dict[str, float],
    full_score: float,
    zero_score: float,
) -> None:
    score = float(metric["score"])
    rows.append(
        {
            "eval_mode": mode,
            "eval_param": param,
            "score": score,
            "loss": float(metric["loss"]),
            "delta_vs_full": score - full_score,
            "gain_vs_zero": score - zero_score,
        }
    )


def evaluate_component_metric(
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


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt_args, metadata = load_checkpoint_model(Path(args.checkpoint), state=args.state, device=device)

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
        raise ValueError("checkpoint/data grid side mismatch")
    _, test_x = normalize_patches(train_x, test_x)

    eval_args = argparse.Namespace(
        eval_batch_size=args.eval_batch_size or int(ckpt_args.get("eval_batch_size", 2048)),
        amp=args.amp or ckpt_args.get("amp", "bf16"),
        mask_rate=args.mask_rate if args.mask_rate is not None else float(ckpt_args.get("mask_rate", 0.35)),
    )
    eval_seed = args.eval_seed if args.eval_seed is not None else int(metadata.get("seed", ckpt_args.get("seed", 426))) + 3000
    rows: list[dict[str, object]] = []
    full_metric = evaluate_component_metric(model, test_x, test_y, task=task, args=eval_args, seed=eval_seed)
    zero_metric = evaluate_component_metric(
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
    add_metric_row(rows, mode="full", param="", metric=full_metric, full_score=full_score, zero_score=zero_score)
    add_metric_row(rows, mode="zero_bias", param="", metric=zero_metric, full_score=full_score, zero_score=zero_score)

    for radius in parse_float_list(args.radial_truncate_radii):
        metric = evaluate_component_metric(
            model,
            test_x,
            test_y,
            task=task,
            args=eval_args,
            seed=eval_seed,
            bias_overrides=eval_control_bias_overrides(
                model,
                mode="radial_truncate",
                holdout_radius=int(metadata.get("holdout_radius", 4)),
                radial_keep_max=radius,
            ),
        )
        add_metric_row(rows, mode="radial_truncate", param=f"r<={radius:g}", metric=metric, full_score=full_score, zero_score=zero_score)

    for radius in parse_float_list(args.layer_radii):
        for layer in parse_int_list(args.layers):
            for mode in ("layer_radial_ablate", "layer_radial_keep"):
                metric = evaluate_component_metric(
                    model,
                    test_x,
                    test_y,
                    task=task,
                    args=eval_args,
                    seed=eval_seed,
                    bias_overrides=component_radial_bias_overrides(model, mode=mode, radius=radius, layer=layer),
                )
                add_metric_row(
                    rows,
                    mode=mode,
                    param=f"layer={layer},r<={radius:g}",
                    metric=metric,
                    full_score=full_score,
                    zero_score=zero_score,
                )

    if args.individual_head_radii:
        for radius in parse_float_list(args.individual_head_radii):
            for layer in parse_int_list(args.head_layers):
                for head in range(len(model.blocks[layer].coeff)):
                    for mode in ("head_radial_ablate", "head_radial_keep"):
                        metric = evaluate_component_metric(
                            model,
                            test_x,
                            test_y,
                            task=task,
                            args=eval_args,
                            seed=eval_seed,
                            bias_overrides=component_radial_bias_overrides(model, mode=mode, radius=radius, layer=layer, head=head),
                        )
                        add_metric_row(
                            rows,
                            mode=mode,
                            param=f"layer={layer},head={head},r<={radius:g}",
                            metric=metric,
                            full_score=full_score,
                            zero_score=zero_score,
                        )

    for radius in parse_float_list(args.head_subset_radii):
        for label, components in parse_head_subset_spec(args.head_subsets):
            for mode in ("head_subset_ablate", "head_subset_keep"):
                metric = evaluate_component_metric(
                    model,
                    test_x,
                    test_y,
                    task=task,
                    args=eval_args,
                    seed=eval_seed,
                    bias_overrides=subset_radial_bias_overrides(model, components=components, radius=radius, mode=mode),
                )
                add_metric_row(
                    rows,
                    mode=mode,
                    param=f"{label},r<={radius:g}",
                    metric=metric,
                    full_score=full_score,
                    zero_score=zero_score,
                )

    write_csv(output_dir / "component_interventions.csv", rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "state": args.state,
        "dataset": dataset,
        "task": task,
        "device": str(device),
        "eval_seed": eval_seed,
        "mask_rate": eval_args.mask_rate,
        "full_score": full_score,
        "zero_score": zero_score,
        "rows": len(rows),
        "output_csv": str(output_dir / "component_interventions.csv"),
        "metadata": metadata,
    }
    (output_dir / "component_interventions_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval V4 component radial interventions from a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state", choices=["final", "best"], default="final")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", choices=["classification", "reconstruction"], default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--amp", choices=["none", "bf16", "fp16"], default=None)
    parser.add_argument("--mask-rate", type=float, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--radial-truncate-radii", type=str, default="2,4")
    parser.add_argument("--layer-radii", type=str, default="2,4")
    parser.add_argument("--layers", type=str, default="0")
    parser.add_argument("--individual-head-radii", type=str, default="4")
    parser.add_argument("--head-layers", type=str, default="0")
    parser.add_argument("--head-subset-radii", type=str, default="2,4")
    parser.add_argument("--head-subsets", type=str, default="0:3+6;0:3+6+1;0:3;0:6;0:1")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
