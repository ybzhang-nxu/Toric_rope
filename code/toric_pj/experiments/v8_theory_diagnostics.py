from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from toric_pj.diagnostics.basis_projection import default_device
from toric_pj.diagnostics.relative_table_geometry import (
    EPS,
    aggregate_geometry_rows,
    geometry_rows,
    load_bias_npz,
    write_csv,
)


PRESET_INPUTS = {
    "cross": [
        "results/v7_cifar100_recon_relative_10k_3seed",
        "results/v7_cifar100_recon_dct_top110_10k_3seed",
        "results/v7_cifar100_recon_toric_pj_top220_10k_3seed",
        "results/v8_cifar100_recon_mixed_residual_dct32_10k_3seed",
        "results/v7_cifar10_recon_top110_30k_3seed",
        "results/v7_cifar10_recon_mixed_residual_dct32_10k_3seed",
        "results/v7_svhn_recon_relative_10k_3seed",
        "results/v7_svhn_recon_basis_10k_3seed_from_relative10k_s426",
    ],
    "tiny-stl": [
        "results/v8_tinyimagenet_relative_10k_3seed",
        "results/v8_tinyimagenet_dct_top110_10k_3seed_from_relative10k_s426",
        "results/v8_tinyimagenet_toric_pj_top220_10k_3seed_from_relative10k_s426",
        "results/v8_tinyimagenet_mixed_residual_dct32_trainradial_r4_10k_3seed_from_relative10k_s426",
        "results/v8_tinyimagenet_compact_smoke_2k_from_relative_s426",
        "results/v8_tinyimagenet_mixed_trainradial_r4_smoke_2k_from_relative_s426",
        "results/v7_stl10_recon_relative_smoke_2k",
        "results/v7_stl10_recon_dct_top110_smoke_2k_from_relative_s426",
        "results/v7_stl10_recon_mixed_residual_dct32_smoke_2k_from_relative_s426",
        "results/v7_stl10_recon_mixed_trainradial_r4_10k_3seed_from_relative_s426",
    ],
    "tiny": [
        "results/v8_tinyimagenet_relative_10k_3seed",
        "results/v8_tinyimagenet_dct_top110_10k_3seed_from_relative10k_s426",
        "results/v8_tinyimagenet_toric_pj_top220_10k_3seed_from_relative10k_s426",
        "results/v8_tinyimagenet_mixed_residual_dct32_trainradial_r4_10k_3seed_from_relative10k_s426",
        "results/v8_tinyimagenet_compact_smoke_2k_from_relative_s426",
        "results/v8_tinyimagenet_mixed_trainradial_r4_smoke_2k_from_relative_s426",
    ],
    "stl": [
        "results/v7_stl10_recon_relative_smoke_2k",
        "results/v7_stl10_recon_dct_top110_smoke_2k_from_relative_s426",
        "results/v7_stl10_recon_mixed_residual_dct32_smoke_2k_from_relative_s426",
        "results/v7_stl10_recon_mixed_trainradial_r4_10k_3seed_from_relative_s426",
    ],
}
PRESET_INPUTS["all"] = PRESET_INPUTS["cross"] + PRESET_INPUTS["tiny-stl"]

PERFORMANCE_TABLES = [
    "results/v7_strong_dataset_summary/v7_main_results.csv",
    "results/v7_strong_dataset_summary/v7_control_results.csv",
    "results/v7_cifar100_main_summary.csv",
    "results/v8_cifar100_mixed_residual_dct32_summary.csv",
    "results/v8_tinyimagenet_10k_summary/tinyimagenet_10k_anchor_compare.csv",
    "results/v8_tinyimagenet_smoke_summary/tinyimagenet_v8_c1_smoke_summary.csv",
    "results/v7_stl10_recon_trainradial_r4_anchor_summary/summary.csv",
    "results/v7_stl10_recon_stabilization_summary/stl10_stabilization_summary.csv",
]


@dataclass
class BiasRecord:
    path: Path
    source_dir: str
    dataset: str
    task: str
    basis: str
    seed: int
    steps: int
    train_radial_visible_radius: int
    label: str


def collect_inputs(paths: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for item in paths:
        path = Path(item)
        if not path.exists():
            continue
        if path.is_dir():
            out.extend(sorted(path.rglob("bias_tables.npz")))
        elif path.name == "bias_tables.npz":
            out.append(path)
        else:
            raise ValueError(f"expected a bias_tables.npz file or directory: {path}")
    unique = sorted(dict.fromkeys(out))
    if not unique:
        raise ValueError("no bias_tables.npz inputs found")
    return unique


def source_dir_for(path: Path) -> str:
    parts = list(path.parts)
    if "bias_exports" in parts:
        idx = parts.index("bias_exports")
        return str(Path(*parts[:idx]))
    return str(path.parent)


def parse_run_name(path: Path) -> dict[str, object]:
    name = path.parent.name
    seed_match = re.search(r"_seed(\d+)", name)
    steps_match = re.search(r"_steps(\d+)", name)
    train_r_match = re.search(r"_trainr(\d+)", name)
    dataset = name.split("_reconstruction_", 1)[0] if "_reconstruction_" in name else "unknown"
    basis = name.split("_reconstruction_", 1)[1] if "_reconstruction_" in name else name
    basis = re.sub(r"_seed\d+.*$", "", basis)
    return {
        "dataset": dataset,
        "basis": basis,
        "seed": int(seed_match.group(1)) if seed_match else -1,
        "steps": int(steps_match.group(1)) if steps_match else -1,
        "train_radial_visible_radius": int(train_r_match.group(1)) if train_r_match else -1,
    }


def normalize_basis(value: str) -> str:
    value = str(value)
    value = value.replace("_trainradial_r4", "")
    value = value.replace("_train_radial_r4", "")
    return value


def infer_record(path: Path, metadata: dict[str, object]) -> BiasRecord:
    parsed = parse_run_name(path)
    dataset = str(metadata.get("dataset") or parsed["dataset"])
    basis = str(metadata.get("basis") or parsed["basis"])
    seed = int(metadata.get("seed", parsed["seed"]))
    steps = int(metadata.get("steps", parsed["steps"]))
    train_r = int(metadata.get("train_radial_visible_radius", parsed["train_radial_visible_radius"]))
    return BiasRecord(
        path=path,
        source_dir=source_dir_for(path),
        dataset=dataset,
        task=str(metadata.get("task", "reconstruction")),
        basis=normalize_basis(basis),
        seed=seed,
        steps=steps,
        train_radial_visible_radius=train_r,
        label=path.parent.name,
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def as_int(value: object, default: int = -1) -> int:
    val = as_float(value, float("nan"))
    return int(val) if math.isfinite(val) else default


def load_performance_rows(paths: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in paths:
        path = Path(item)
        for row in read_csv_rows(path):
            basis = normalize_basis(str(row.get("basis", "")))
            if not basis:
                continue
            perf = {
                "perf_source": str(path),
                "perf_label": row.get("label", basis),
                "dataset": row.get("dataset", "tiny-imagenet" if "tinyimagenet" in str(path) else ""),
                "basis": basis,
                "seed": as_int(row.get("seed"), -1),
                "steps": as_int(row.get("steps"), -1),
                "train_radial_visible_radius": as_int(row.get("train_radial_visible_radius"), -1),
                "output_dir": row.get("output_dir", ""),
                "final": as_float(row.get("final", row.get("final_mean"))),
                "visible": as_float(row.get("visible", row.get("visible_mean"))),
                "zero": as_float(row.get("zero", row.get("zero_mean"))),
                "full_minus_visible": as_float(row.get("full_minus_visible", row.get("full_minus_visible_mean"))),
                "final_minus_zero": as_float(row.get("final_minus_zero", row.get("final_minus_zero_mean", row.get("full_minus_zero_mean")))),
                "heldout_clamp": as_float(row.get("heldout_clamp")),
                "radial_r4": as_float(row.get("radial_r4", row.get("radial_truncate_r4"))),
                "radial_r5": as_float(row.get("radial_r5")),
                "radial_band_6_9": as_float(row.get("radial_band_6_9")),
                "heldout_rms": as_float(row.get("heldout_rms_mean", row.get("heldout_rms"))),
            }
            rows.append(perf)
    return rows


def perf_match_score(record: BiasRecord, row: dict[str, object]) -> int:
    score = 0
    row_dataset = str(row.get("dataset", ""))
    if row_dataset and row_dataset != record.dataset:
        return -1
    if normalize_basis(str(row.get("basis", ""))) != normalize_basis(record.basis):
        return -1
    score += 4
    output_dir = str(row.get("output_dir", ""))
    if output_dir and output_dir == record.source_dir:
        score += 12
    row_steps = int(row.get("steps", -1))
    if row_steps == record.steps:
        score += 6
    elif row_steps != -1 and record.steps != -1:
        score -= 4
    row_seed = int(row.get("seed", -1))
    if row_seed == record.seed:
        score += 5
    elif row_seed != -1 and record.seed != -1:
        score -= 2
    row_train_r = int(row.get("train_radial_visible_radius", -1))
    if row_train_r == record.train_radial_visible_radius:
        score += 5
    elif row_train_r != -1 and record.train_radial_visible_radius != -1:
        score -= 4
    return score


def match_performance(record: BiasRecord, rows: list[dict[str, object]]) -> dict[str, object]:
    best: tuple[int, dict[str, object] | None] = (-1, None)
    for row in rows:
        score = perf_match_score(record, row)
        if score > best[0]:
            best = (score, row)
    if best[1] is None or best[0] < 4:
        return {}
    return dict(best[1])


def load_control_metrics(record: BiasRecord) -> dict[str, object]:
    path = Path(record.source_dir) / "offset_holdout_eval_controls_aggregate.csv"
    if not path.exists():
        return {}
    controls: dict[str, object] = {"control_source": str(path)}
    for row in read_csv_rows(path):
        if normalize_basis(str(row.get("basis", ""))) != normalize_basis(record.basis):
            continue
        row_train_r = as_int(row.get("train_radial_visible_radius"), -1)
        if row_train_r != record.train_radial_visible_radius:
            continue
        score = as_float(row.get("score_mean", row.get("score")))
        if not math.isfinite(score):
            continue
        mode = str(row.get("eval_mode", ""))
        param = str(row.get("eval_param", ""))
        if mode == "full":
            controls["final"] = score
        elif mode == "visible_only":
            controls["visible"] = score
        elif mode == "zero_bias":
            controls["zero"] = score
        elif mode == "heldout_clamp":
            controls["heldout_clamp"] = score
        elif mode == "radial_truncate" and param == "r<=4":
            controls["radial_r4"] = score
        elif mode == "radial_truncate" and param == "r<=5":
            controls["radial_r5"] = score
        elif mode == "radial_band" and param == "6<=r<9":
            controls["radial_band_6_9"] = score
    if "final" in controls and "visible" in controls:
        controls["full_minus_visible"] = as_float(controls["final"]) - as_float(controls["visible"])
    if "final" in controls and "zero" in controls:
        controls["final_minus_zero"] = as_float(controls["final"]) - as_float(controls["zero"])
    return controls


def merged_metric(primary: dict[str, object], secondary: dict[str, object], key: str) -> object:
    secondary_value = secondary.get(key, float("nan"))
    if math.isfinite(as_float(secondary_value)):
        return secondary_value
    return primary.get(key, float("nan"))


def mean_numeric(rows: list[dict[str, object]], *, prefix: str = "") -> dict[str, float]:
    if not rows:
        return {}
    keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float, np.integer, np.floating)) and key not in {"seed", "layer", "head", "n"}
    ]
    out: dict[str, float] = {}
    for key in keys:
        vals = np.array([float(row[key]) for row in rows if math.isfinite(float(row[key]))], dtype=np.float64)
        if vals.size:
            out[f"{prefix}{key}_mean"] = float(vals.mean())
            out[f"{prefix}{key}_std"] = float(vals.std())
    return out


def radial_masks(dx_values: torch.Tensor, dy_values: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    xx, yy = torch.meshgrid(dx_values, dy_values, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    masks = {
        "r_le_2": radius <= 2.0,
        "r_le_4": radius <= 4.0,
        "r_4_6": (radius > 4.0) & (radius <= 6.0),
        "r_6_9": (radius > 6.0) & (radius < 9.0),
        "r_ge_9": radius >= 9.0,
    }
    return radius, masks


def radial_summary(tables: torch.Tensor, dx_values: torch.Tensor, dy_values: torch.Tensor) -> tuple[dict[str, float], list[dict[str, object]]]:
    centered = tables - tables.mean(dim=(-2, -1), keepdim=True)
    energy = centered.square()
    total = energy.sum(dim=(-2, -1)).clamp_min(EPS)
    radius, masks = radial_masks(dx_values.to(tables.device), dy_values.to(tables.device))
    rows: list[dict[str, object]] = []
    out: dict[str, float] = {}
    for name, mask in masks.items():
        share = energy[..., mask].sum(dim=-1) / total
        out[f"energy_share_{name}_mean"] = float(share.mean().detach().cpu())
        out[f"energy_share_{name}_std"] = float(share.std(unbiased=False).detach().cpu())
    near = energy[..., masks["r_le_4"]].sum(dim=-1)
    far = energy[..., masks["r_6_9"]].sum(dim=-1)
    out["far_to_near_energy_mean"] = float((far / near.clamp_min(EPS)).mean().detach().cpu())
    centroid = (energy * radius).sum(dim=(-2, -1)) / total
    out["radial_energy_centroid_mean"] = float(centroid.mean().detach().cpu())
    max_radius = int(torch.ceil(radius.max()).item())
    for r_int in range(max_radius + 1):
        mask = (radius >= float(r_int)) & (radius < float(r_int + 1))
        if not bool(mask.any()):
            continue
        share = energy[..., mask].sum(dim=-1) / total
        rows.append(
            {
                "radius_bin": r_int,
                "radius_lo": float(r_int),
                "radius_hi": float(r_int + 1),
                "energy_share_mean": float(share.mean().detach().cpu()),
                "energy_share_std": float(share.std(unbiased=False).detach().cpu()),
            }
        )
    return out, rows


def spectral_curvature_summary(tables: torch.Tensor) -> dict[str, float]:
    centered = tables - tables.mean(dim=(-2, -1), keepdim=True)
    fft = torch.fft.fft2(centered)
    power = fft.abs().square()
    power[..., 0, 0] = 0.0
    size = tables.shape[-1]
    freq = torch.fft.fftfreq(size, device=tables.device, dtype=tables.dtype) * (2.0 * math.pi)
    wx, wy = torch.meshgrid(freq, freq, indexing="ij")
    lap_eigen = 4.0 - 2.0 * torch.cos(wx) - 2.0 * torch.cos(wy)
    total = power.sum(dim=(-2, -1)).clamp_min(EPS)
    spectral_curv = (lap_eigen.square() * power).sum(dim=(-2, -1)) / total
    spectral_freq = (torch.sqrt(wx.square() + wy.square()) * power).sum(dim=(-2, -1)) / total
    return {
        "spectral_curvature_energy_mean": float(spectral_curv.mean().detach().cpu()),
        "spectral_curvature_energy_std": float(spectral_curv.std(unbiased=False).detach().cpu()),
        "spectral_frequency_centroid_mean": float(spectral_freq.mean().detach().cpu()),
        "spectral_frequency_centroid_std": float(spectral_freq.std(unbiased=False).detach().cpu()),
    }


def analyze_bias(path: Path, *, device: torch.device, topk: int, perf_rows: list[dict[str, object]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    bundle, metadata = load_bias_npz(path, device=device)
    record = infer_record(path, metadata)
    geom = geometry_rows(
        bundle.tables,
        basis=record.basis,
        dataset=record.dataset,
        task=record.task,
        seed=record.seed,
        topk=topk,
    )
    centered_interior = [
        row for row in geom if row.get("gauge") == "centered" and row.get("boundary") == "interior_only"
    ]
    normalized_interior = [
        row for row in geom if row.get("gauge") == "normalized" and row.get("boundary") == "interior_only"
    ]
    radial_metrics, radial_rows = radial_summary(bundle.tables, bundle.dx_values, bundle.dy_values)
    spectral_curv = spectral_curvature_summary(bundle.tables)
    perf = match_performance(record, perf_rows)
    controls = load_control_metrics(record)
    row: dict[str, object] = {
        "dataset": record.dataset,
        "task": record.task,
        "basis": record.basis,
        "seed": record.seed,
        "steps": record.steps,
        "train_radial_visible_radius": record.train_radial_visible_radius,
        "source_dir": record.source_dir,
        "bias_npz": str(path),
        "label": record.label,
        "n_layers": int(bundle.tables.shape[0]),
        "n_heads": int(bundle.tables.shape[1]),
        "table_side": int(bundle.tables.shape[-1]),
    }
    for key in [
        "final",
        "visible",
        "zero",
        "final_minus_zero",
        "full_minus_visible",
        "heldout_clamp",
        "radial_r4",
        "radial_r5",
        "radial_band_6_9",
        "heldout_rms",
    ]:
        row[key] = merged_metric(perf, controls, key)
    row["perf_source"] = perf.get("perf_source", "")
    row["perf_label"] = perf.get("perf_label", "")
    row["control_source"] = controls.get("control_source", "")
    row.update(mean_numeric(centered_interior, prefix="centered_"))
    row.update(mean_numeric(normalized_interior, prefix="normalized_"))
    row.update(radial_metrics)
    row.update(spectral_curv)
    boundary_gap = -as_float(row.get("full_minus_visible"), 0.0)
    row["boundary_pollution"] = max(0.0, boundary_gap)
    row["local_geometry_score"] = as_float(row.get("energy_share_r_le_4_mean"), 0.0) * as_float(
        row.get("centered_obl_ratio_mean"), 0.0
    )
    for radial_row in radial_rows:
        radial_row.update(
            {
                "dataset": record.dataset,
                "basis": record.basis,
                "seed": record.seed,
                "steps": record.steps,
                "train_radial_visible_radius": record.train_radial_visible_radius,
                "source_dir": record.source_dir,
            }
        )
    return row, radial_rows


def aggregate_diagnostic_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["basis"]),
            int(row.get("steps", -1)),
            int(row.get("train_radial_visible_radius", -1)),
            str(row.get("source_dir", "")),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, object]] = []
    skip = {
        "dataset",
        "task",
        "basis",
        "source_dir",
        "bias_npz",
        "label",
        "perf_source",
        "perf_label",
    }
    for (dataset, basis, steps, train_r, source_dir), values in sorted(groups.items()):
        item: dict[str, object] = {
            "dataset": dataset,
            "basis": basis,
            "steps": steps,
            "train_radial_visible_radius": train_r,
            "source_dir": source_dir,
            "family": Path(source_dir).name if source_dir else "",
            "n": len(values),
            "seeds": ",".join(str(int(v["seed"])) for v in sorted(values, key=lambda x: int(x["seed"]))),
        }
        keys = [
            key
            for key, value in values[0].items()
            if key not in skip and isinstance(value, (int, float, np.integer, np.floating))
        ]
        for key in keys:
            vals = np.array([as_float(row.get(key)) for row in values], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                item[f"{key}_mean"] = float(vals.mean())
                item[f"{key}_std"] = float(vals.std())
        out.append(item)
    return out


def plot_bar(rows: list[dict[str, object]], output_dir: Path, *, key: str, filename: str, title: str) -> None:
    plot_rows = [row for row in rows if math.isfinite(as_float(row.get(key)))]
    if not plot_rows:
        return
    labels = [
        f"{row['dataset']}\n{short_basis(str(row['basis']))}\n{row.get('family', '')}\nsteps={int(row['steps'])}, r={int(row['train_radial_visible_radius'])}"
        for row in plot_rows
    ]
    values = [as_float(row.get(key)) for row in plot_rows]
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(plot_rows)), 4.8))
    ax.bar(range(len(plot_rows)), values, color="#3b82f6")
    ax.set_xticks(range(len(plot_rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(key)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def plot_radial_profiles(rows: list[dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return
    groups: dict[tuple[str, str, int, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["basis"]),
            int(row["steps"]),
            int(row["train_radial_visible_radius"]),
            str(row.get("source_dir", "")),
        )
        groups.setdefault(key, []).append(row)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for (dataset, basis, steps, train_r, source_dir), values in sorted(groups.items()):
        by_radius: dict[int, list[float]] = {}
        for row in values:
            by_radius.setdefault(int(row["radius_bin"]), []).append(float(row["energy_share_mean"]))
        xs = sorted(by_radius)
        ys = [float(np.mean(by_radius[x])) for x in xs]
        family = Path(source_dir).name if source_dir else ""
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=3, label=f"{dataset} {short_basis(basis)} {family} s{steps} r{train_r}")
    ax.set_xlabel("relative radius bin")
    ax.set_ylabel("mean centered bias energy share")
    ax.set_title("Radial Energy Profiles")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "radial_energy_profiles.png", dpi=180)
    plt.close(fig)


def short_basis(basis: str) -> str:
    replacements = [
        ("mixed_toric_PJ_R0_top220_residual_dct_top32", "mixed residual-DCT32"),
        ("table_informed_toric_PJ_R0_top220", "Toric/PJ top220"),
        ("relative_2d_table", "relative"),
        ("dct_top110", "DCT top110"),
    ]
    for old, new in replacements:
        basis = basis.replace(old, new)
    return basis


def fmt(value: object, digits: int = 4) -> str:
    val = as_float(value)
    return "nan" if not math.isfinite(val) else f"{val:.{digits}f}"


def write_summary(output_dir: Path, rows: list[dict[str, object]], aggregate: list[dict[str, object]]) -> None:
    lines = [
        "# V8 Theory Diagnostics",
        "",
        "Date: 2026-06-11",
        "",
        "## Readout Table",
        "",
        "| dataset | basis | family | steps | train r | final | visible | full-visible | r<=4 score | far band | obl ratio | mixed Hessian | spectral curvature | r<=4 energy | r6-9 energy |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    short_basis(str(row["basis"])),
                    str(row.get("family", "")),
                    str(int(row["steps"])),
                    str(int(row["train_radial_visible_radius"])),
                    fmt(row.get("final_mean")),
                    fmt(row.get("visible_mean")),
                    fmt(row.get("full_minus_visible_mean")),
                    fmt(row.get("radial_r4_mean")),
                    fmt(row.get("radial_band_6_9_mean")),
                    fmt(row.get("centered_obl_ratio_mean_mean")),
                    fmt(row.get("centered_mixed_ratio_mean_mean")),
                    fmt(row.get("spectral_curvature_energy_mean_mean")),
                    fmt(row.get("energy_share_r_le_4_mean_mean")),
                    fmt(row.get("energy_share_r_6_9_mean_mean")),
                ]
            )
            + " |"
        )
    tiny_toric = find_aggregate(aggregate, "tiny-imagenet", "table_informed_toric_PJ_R0_top220", 10000, -1)
    tiny_mixed = find_aggregate(aggregate, "tiny-imagenet", "mixed_toric_PJ_R0_top220_residual_dct_top32", 10000, 4)
    stl_naive = find_aggregate(aggregate, "stl10", "mixed_toric_PJ_R0_top220_residual_dct_top32", 2000, -1)
    stl_r4 = find_aggregate(aggregate, "stl10", "mixed_toric_PJ_R0_top220_residual_dct_top32", 10000, 4)
    cifar100_toric = find_aggregate(aggregate, "cifar100", "table_informed_toric_PJ_R0_top220", 10000, -1)
    cifar100_mixed = find_aggregate(aggregate, "cifar100", "mixed_toric_PJ_R0_top220_residual_dct_top32", 10000, -1)
    cifar10_toric = find_aggregate(aggregate, "cifar10", "table_informed_toric_PJ_R0_top110", 30000, -1)
    cifar10_mixed = find_aggregate(aggregate, "cifar10", "mixed_toric_PJ_R0_top220_residual_dct_top32", 10000, -1)
    svhn_toric = find_aggregate(aggregate, "svhn", "table_informed_toric_PJ_R0_top220", 10000, -1)
    svhn_mixed = find_aggregate(aggregate, "svhn", "mixed_toric_PJ_R0_top220_residual_dct_top32", 10000, -1)
    lines.extend(["", "## Mechanism Notes", ""])
    if tiny_toric and tiny_mixed:
        lines.extend(
            [
                "- TinyImageNet Toric/PJ and mixed train-radial share high local geometry: "
                + f"Toric/PJ r<=4 score {fmt(tiny_toric.get('radial_r4_mean'))}, "
                + f"mixed r<=4 score {fmt(tiny_mixed.get('radial_r4_mean'))}.",
                "- Mixed train-radial lowers boundary pollution relative to pure Toric/PJ while preserving oblique geometry: "
                + f"full-visible improves from {fmt(tiny_toric.get('full_minus_visible_mean'))} "
                + f"to {fmt(tiny_mixed.get('full_minus_visible_mean'))}.",
                "- The far-band readout matches the causal story: "
                + f"Toric/PJ far band {fmt(tiny_toric.get('radial_band_6_9_mean'))}, "
                + f"mixed far band {fmt(tiny_mixed.get('radial_band_6_9_mean'))}.",
            ]
        )
    if stl_naive and stl_r4:
        lines.extend(
            [
                "- STL10 naive mixed has strong visible/local structure but severe boundary pollution; "
                + f"train-radial r=4 moves full-visible from {fmt(stl_naive.get('full_minus_visible_mean'))} "
                + f"to {fmt(stl_r4.get('full_minus_visible_mean'))}.",
                "- This is the cleanest current bridge from 理论背景v2: local toric/oblique geometry can be useful, "
                "but far-field coordinate policy decides whether full inference survives.",
            ]
        )
    if cifar100_toric and cifar100_mixed:
        lines.extend(
            [
                "- CIFAR100 repeats the Toric/PJ -> mixed rescue pattern at high score: "
                + f"Toric/PJ full-visible {fmt(cifar100_toric.get('full_minus_visible_mean'))}, "
                + f"r<=4 energy {fmt(cifar100_toric.get('energy_share_r_le_4_mean_mean'))}, "
                + f"r6-9 energy {fmt(cifar100_toric.get('energy_share_r_6_9_mean_mean'))}; "
                + f"mixed full-visible {fmt(cifar100_mixed.get('full_minus_visible_mean'))}, "
                + f"r<=4 energy {fmt(cifar100_mixed.get('energy_share_r_le_4_mean_mean'))}, "
                + f"r6-9 energy {fmt(cifar100_mixed.get('energy_share_r_6_9_mean_mean'))}.",
            ]
        )
    if cifar10_toric and cifar10_mixed:
        lines.extend(
            [
                "- CIFAR10 is a strong boundary-failure/rescue contrast: "
                + f"Toric/PJ top110 full-visible {fmt(cifar10_toric.get('full_minus_visible_mean'))}, "
                + f"far band {fmt(cifar10_toric.get('radial_band_6_9_mean'))}, "
                + f"r6-9 energy {fmt(cifar10_toric.get('energy_share_r_6_9_mean_mean'))}; "
                + f"mixed full-visible {fmt(cifar10_mixed.get('full_minus_visible_mean'))}, "
                + f"far band {fmt(cifar10_mixed.get('radial_band_6_9_mean'))}, "
                + f"r6-9 energy {fmt(cifar10_mixed.get('energy_share_r_6_9_mean_mean'))}.",
            ]
        )
    if svhn_toric and svhn_mixed:
        lines.extend(
            [
                "- SVHN is the most extreme cross-dataset case: "
                + f"Toric/PJ has visible {fmt(svhn_toric.get('visible_mean'))} but full {fmt(svhn_toric.get('final_mean'))}, "
                + f"with r<=4 energy {fmt(svhn_toric.get('energy_share_r_le_4_mean_mean'))} "
                + f"and r6-9 energy {fmt(svhn_toric.get('energy_share_r_6_9_mean_mean'))}; "
                + f"mixed recovers full {fmt(svhn_mixed.get('final_mean'))} while moving energy back toward the local chart "
                + f"(r<=4 {fmt(svhn_mixed.get('energy_share_r_le_4_mean_mean'))}, "
                + f"r6-9 {fmt(svhn_mixed.get('energy_share_r_6_9_mean_mean'))}).",
            ]
        )
    if (cifar100_toric and cifar100_mixed) or (cifar10_toric and cifar10_mixed) or (svhn_toric and svhn_mixed):
        lines.extend(
            [
                "- Cross-dataset readout: high oblique/axial-residual geometry is common in both successes and failures; "
                "the decisive diagnostic is whether the learned potential concentrates useful mass near r<=4 or leaks into the r6-9 shell.",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `diagnostic_metrics.csv`: one row per bias export.",
            "- `diagnostic_aggregate.csv`: grouped by dataset/basis/steps/train-radial/source directory.",
            "- `radial_energy_profiles.csv`: per-radius centered-bias energy profile.",
            "- `mixed_hessian_comparison.png`.",
            "- `axial_residual_comparison.png`.",
            "- `spectral_curvature_comparison.png`.",
            "- `radial_energy_profiles.png`.",
        ]
    )
    (output_dir / "diagnostic_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_aggregate(
    rows: list[dict[str, object]],
    dataset: str,
    basis: str,
    steps: int,
    train_r: int,
) -> dict[str, object] | None:
    for row in rows:
        if (
            row.get("dataset") == dataset
            and row.get("basis") == basis
            and int(row.get("steps", -1)) == steps
            and int(row.get("train_radial_visible_radius", -1)) == train_r
        ):
            return row
    return None


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = list(args.input or [])
    if args.preset:
        inputs.extend(PRESET_INPUTS[args.preset])
    paths = collect_inputs(inputs)
    perf_rows = load_performance_rows(PERFORMANCE_TABLES + list(args.performance_csv or []))

    metric_rows: list[dict[str, object]] = []
    radial_rows: list[dict[str, object]] = []
    for path in paths:
        row, radial = analyze_bias(path, device=device, topk=args.topk, perf_rows=perf_rows)
        metric_rows.append(row)
        radial_rows.extend(radial)

    aggregate_rows = aggregate_diagnostic_rows(metric_rows)
    write_csv(output_dir / "diagnostic_metrics.csv", metric_rows)
    write_csv(output_dir / "diagnostic_aggregate.csv", aggregate_rows)
    write_csv(output_dir / "radial_energy_profiles.csv", radial_rows)
    plot_bar(
        aggregate_rows,
        output_dir,
        key="centered_mixed_ratio_mean_mean",
        filename="mixed_hessian_comparison.png",
        title="Mixed Hessian Ratio",
    )
    plot_bar(
        aggregate_rows,
        output_dir,
        key="centered_obl_ratio_mean_mean",
        filename="axial_residual_comparison.png",
        title="Axial Residual / Oblique Energy Ratio",
    )
    plot_bar(
        aggregate_rows,
        output_dir,
        key="spectral_curvature_energy_mean_mean",
        filename="spectral_curvature_comparison.png",
        title="Spectral Curvature Energy",
    )
    plot_radial_profiles(radial_rows, output_dir)
    write_summary(output_dir, metric_rows, aggregate_rows)
    summary = {
        "preset": args.preset,
        "inputs": [str(path) for path in paths],
        "output_dir": str(output_dir),
        "num_metric_rows": len(metric_rows),
        "num_aggregate_rows": len(aggregate_rows),
        "num_radial_rows": len(radial_rows),
        "device": str(device),
    }
    (output_dir / "theory_diagnostics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V8 theory diagnostics for exported 2D relative bias tables.")
    parser.add_argument("--input", nargs="*", default=[], help="bias_tables.npz files or directories.")
    parser.add_argument("--preset", choices=sorted(PRESET_INPUTS), default=None)
    parser.add_argument("--performance-csv", nargs="*", default=[], help="additional performance summary CSV files.")
    parser.add_argument("--output-dir", default="results/v8_theory_diagnostics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
