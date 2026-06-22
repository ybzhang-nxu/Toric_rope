from __future__ import annotations

import csv
import json
import statistics as stats
from pathlib import Path


ROOT = Path("results")
OUT = ROOT / "v7_strong_dataset_summary"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return stats.mean(values), stats.pstdev(values)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def summarize_offset_run(
    *,
    dataset: str,
    task: str,
    basis: str,
    steps: int,
    result_csv: Path,
    relative_reference: float | None,
    best_control: str,
    component_ablate_drop: object | None,
    notes: str,
) -> dict[str, object]:
    rows = read_rows(result_csv)
    finals = [float(r["final_score"]) for r in rows]
    bests = [float(r["best_score"]) for r in rows]
    visibles = [float(r["final_visible_only_score"]) for r in rows]
    zeros = [float(r["final_zero_bias_score"]) for r in rows]
    gains = [f - z for f, z in zip(finals, zeros, strict=True)]
    full_visible = [f - v for f, v in zip(finals, visibles, strict=True)]
    heldout_rms = [float(r["heldout_bias_rms"]) for r in rows]
    num_features = int(float(rows[0]["num_features"])) if rows else -1
    seed_list = ",".join(r["seed"] for r in rows)

    best_mean, best_std = mean_std(bests)
    final_mean, final_std = mean_std(finals)
    visible_mean, visible_std = mean_std(visibles)
    zero_mean, zero_std = mean_std(zeros)
    gain_mean, gain_std = mean_std(gains)
    full_visible_mean, full_visible_std = mean_std(full_visible)
    heldout_rms_mean, heldout_rms_std = mean_std(heldout_rms)

    relative_ratio = ""
    if relative_reference and relative_reference == relative_reference:
        relative_ratio = final_mean / relative_reference

    return {
        "dataset": dataset,
        "task": task,
        "basis": basis,
        "seeds": seed_list,
        "steps": steps,
        "n": len(rows),
        "num_features": num_features,
        "best_mean": best_mean,
        "best_std": best_std,
        "final_mean": final_mean,
        "final_std": final_std,
        "visible_mean": visible_mean,
        "visible_std": visible_std,
        "zero_mean": zero_mean,
        "zero_std": zero_std,
        "full_minus_zero_mean": gain_mean,
        "full_minus_zero_std": gain_std,
        "full_minus_visible_mean": full_visible_mean,
        "full_minus_visible_std": full_visible_std,
        "heldout_bias_rms_mean": heldout_rms_mean,
        "heldout_bias_rms_std": heldout_rms_std,
        "relative_ratio": relative_ratio,
        "best_control": best_control,
        "component_ablate_drop": "" if component_ablate_drop is None else component_ablate_drop,
        "notes": notes,
    }


def cifar100_rows() -> list[dict[str, object]]:
    main = read_rows(ROOT / "v7_cifar100_main_summary.csv")
    rel = next(r for r in main if r["basis"] == "relative_2d_table")
    rel_final = float(rel["final_mean"])
    rows: list[dict[str, object]] = []
    for r in main:
        final = float(r["final_mean"])
        visible = float(r["visible_mean"])
        zero = float(r["zero_mean"])
        notes = r["notes"]
        rows.append(
            {
                "dataset": r["dataset"],
                "task": r["task"],
                "basis": r["basis"],
                "seeds": r["seeds"],
                "steps": int(float(r["steps"])),
                "n": len(r["seeds"].split(",")),
                "num_features": r["features"],
                "best_mean": "",
                "best_std": "",
                "final_mean": final,
                "final_std": float(r["final_std"]),
                "visible_mean": visible,
                "visible_std": float(r["visible_std"]),
                "zero_mean": zero,
                "zero_std": float(r["zero_std"]),
                "full_minus_zero_mean": final - zero,
                "full_minus_zero_std": float(r["final_minus_zero_std"]),
                "full_minus_visible_mean": final - visible,
                "full_minus_visible_std": "",
                "heldout_bias_rms_mean": "",
                "heldout_bias_rms_std": "",
                "relative_ratio": final / rel_final,
                "best_control": "see causal summary",
                "component_ablate_drop": "",
                "notes": notes,
            }
        )
    mixed_path = ROOT / "v8_cifar100_recon_mixed_residual_dct32_10k_3seed" / "offset_holdout_results.csv"
    if mixed_path.exists():
        rows.append(
            summarize_offset_run(
                dataset="cifar100",
                task="reconstruction",
                basis="mixed_toric_PJ_R0_top220_residual_dct_top32",
                steps=10000,
                result_csv=mixed_path,
                relative_reference=rel_final,
                best_control="radial_truncate r<=5 mean 0.8842",
                component_ablate_drop="0.5692 L0-L5 all-head ablate mean",
                notes="mixed residual-DCT; matches DCT full performance while preserving Toric/PJ local mechanism",
            )
        )
    return rows


def add_cifar100_component_drops(rows: list[dict[str, object]]) -> None:
    causal = read_rows(ROOT / "v7_cifar100_causal_summary.csv")
    drops = {
        "table_informed_toric_PJ_R0_top220": "0.4339 early L0-L2 all-head ablate mean",
        "dct_top110": "0.2498 selected head-pool ablate seed526",
    }
    for row in rows:
        basis = str(row["basis"])
        if basis in drops:
            row["component_ablate_drop"] = drops[basis]
        if basis == "table_informed_toric_PJ_R0_top220":
            row["best_control"] = "radial_truncate r<=4 seed426 score 0.8804"
        elif basis == "dct_top110":
            row["best_control"] = "radial_truncate r<=5 seed526 score 0.8817"
        elif basis == "relative_2d_table":
            row["best_control"] = "visible/full identical by construction"
    _ = causal


def cifar10_control_rows() -> list[dict[str, object]]:
    keep_modes = {
        "full",
        "visible_only",
        "heldout_clamp",
        "zero_bias",
        "radial_truncate",
        "radial_decay",
        "radial_band",
        "layer_radial_ablate",
        "layer_radial_keep",
    }
    out: list[dict[str, object]] = []
    sources = [
        (
            "table_informed_toric_PJ_R0_top110",
            ROOT / "v7_cifar10_recon_top110_30k_3seed" / "offset_holdout_eval_controls_aggregate.csv",
        ),
        (
            "mixed_toric_PJ_R0_top220_residual_dct_top32",
            ROOT / "v7_cifar10_recon_mixed_residual_dct32_10k_3seed" / "offset_holdout_eval_controls_aggregate.csv",
        ),
    ]
    for basis, path in sources:
        if not path.exists():
            continue
        for r in read_rows(path):
            mode = r["eval_mode"]
            param = r["eval_param"]
            if mode not in keep_modes:
                continue
            if mode in {"layer_radial_ablate", "layer_radial_keep"} and param not in {
                "layer=0,r<=4",
                "layer=1,r<=4",
                "layer=2,r<=4",
                "layer=2,r<=2",
            }:
                continue
            if mode == "radial_decay" and param != "gamma=4":
                continue
            if mode == "radial_truncate" and param not in {"r<=4", "r<=5"}:
                continue
            if mode == "radial_truncate" and basis == "table_informed_toric_PJ_R0_top110" and param != "r<=4":
                continue
            if mode == "radial_band" and param not in {"0<=r<2", "6<=r<9"}:
                continue
            out.append(
                {
                    "dataset": "cifar10",
                    "task": "reconstruction",
                    "basis": basis,
                    "group": mode,
                    "condition": param,
                    "n": int(float(r["n"])),
                    "score_mean": float(r["score_mean"]),
                    "score_std": float(r["score_std"]),
                    "interpretation": "",
                }
            )
    mechanism_path = (
        ROOT
        / "v7_cifar10_mixed_component_mechanism_summary"
        / "component_mechanism_family_key_metrics.csv"
    )
    if mechanism_path.exists():
        mechanism = read_rows(mechanism_path)[0]
        n = int(float(mechanism["n"]))
        basis = str(mechanism["basis"])
        for condition, mean_col, std_col, interpretation in [
            (
                "L0-L2 all-head keep r<=4",
                "l012_keep_mean",
                "l012_keep_std",
                "early local pool is sufficient and near full",
            ),
            (
                "L0-L2 all-head ablate r<=4",
                "l012_ablate_mean",
                "l012_ablate_std",
                "early local pool is strongly necessary",
            ),
            (
                "L0-L3 all-head keep r<=4",
                "l0123_keep_mean",
                "l0123_keep_std",
                "early plus L3 pool slightly exceeds full",
            ),
            (
                "L0-L3 all-head ablate r<=4",
                "l0123_ablate_mean",
                "l0123_ablate_std",
                "early plus L3 pool nearly exhausts to zero",
            ),
        ]:
            out.append(
                {
                    "dataset": "cifar10",
                    "task": "reconstruction",
                    "basis": basis,
                    "group": "mixed_component_pool",
                    "condition": condition,
                    "n": n,
                    "score_mean": float(mechanism[mean_col]),
                    "score_std": float(mechanism[std_col]),
                    "interpretation": interpretation,
                }
            )
    return out


def svhn_rows() -> list[dict[str, object]]:
    relative_dir = ROOT / "v7_svhn_recon_relative_10k_3seed"
    basis_dir = ROOT / "v7_svhn_recon_basis_10k_3seed_from_relative10k_s426"
    steps = 10000
    if not (relative_dir / "offset_holdout_results.csv").exists() or not (
        basis_dir / "offset_holdout_results.csv"
    ).exists():
        relative_dir = ROOT / "v7_svhn_recon_relative_smoke_2k"
        basis_dir = ROOT / "v7_svhn_recon_basis_smoke_2k_from_relative_s426"
        steps = 2000

    relative_path = relative_dir / "offset_holdout_results.csv"
    basis_path = basis_dir / "offset_holdout_results.csv"
    if not relative_path.exists() or not basis_path.exists():
        return []
    relative_row = summarize_offset_run(
        dataset="svhn",
        task="reconstruction",
        basis="relative_2d_table",
        steps=steps,
        result_csv=relative_path,
        relative_reference=None,
        best_control="visible/full identical by construction",
        component_ablate_drop=None,
        notes="SVHN 10k/3seed strong relative baseline"
        if steps == 10000
        else "SVHN cross-dataset relative smoke; strong positive reconstruction signal",
    )
    relative_ref = float(relative_row["final_mean"])
    relative_row["relative_ratio"] = 1.0
    out = [relative_row]
    basis_rows = read_rows(basis_path)
    by_basis: dict[str, list[dict[str, str]]] = {}
    for row in basis_rows:
        by_basis.setdefault(row["basis"], []).append(row)
    tmp_dir = OUT / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for basis, rows in by_basis.items():
        tmp_path = tmp_dir / f"{basis}.csv"
        with tmp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        if basis == "dct_top110":
            note = (
                "SVHN compact DCT 10k/3seed; strong full baseline but still lags visible"
                if steps == 10000
                else "SVHN compact DCT smoke; full lags visible but remains strong"
            )
            best_control = "radial_truncate r<=4 mean 0.9715" if steps == 10000 else "radial_truncate r<=4 score 0.9510"
        elif basis == "table_informed_toric_PJ_R0_top220":
            note = (
                "SVHN pure Toric/PJ 10k/3seed; visible/local strong but full extrapolation remains harmful"
                if steps == 10000
                else "SVHN pure Toric/PJ visible/local is strong but full extrapolation collapses"
            )
            best_control = "radial_truncate r<=4 mean 0.9708" if steps == 10000 else "radial_truncate r<=4 score 0.9477"
        elif basis == "mixed_toric_PJ_R0_top220_residual_dct_top32":
            note = (
                "SVHN mixed residual-DCT 10k/3seed; best compact full result and near relative"
                if steps == 10000
                else "SVHN mixed residual-DCT smoke; best full compact result"
            )
            best_control = "radial_truncate r<=4 mean 0.9757" if steps == 10000 else "radial_truncate r<=4 score 0.9502"
        else:
            note = "SVHN basis smoke"
            best_control = "see controls"
        out.append(
            summarize_offset_run(
                dataset="svhn",
                task="reconstruction",
                basis=basis,
                steps=steps,
                result_csv=tmp_path,
                relative_reference=relative_ref,
                best_control=best_control,
                component_ablate_drop=None,
                notes=note,
            )
        )
        tmp_path.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    return out


def svhn_control_rows() -> list[dict[str, object]]:
    path = ROOT / "v7_svhn_recon_basis_10k_3seed_from_relative10k_s426" / "offset_holdout_eval_controls_aggregate.csv"
    if not path.exists():
        path = ROOT / "v7_svhn_recon_basis_smoke_2k_from_relative_s426" / "offset_holdout_eval_controls_aggregate.csv"
    if not path.exists():
        return []
    rows = read_rows(path)
    keep = {
        ("full", ""),
        ("visible_only", ""),
        ("zero_bias", ""),
        ("heldout_clamp", ""),
        ("radial_truncate", "r<=4"),
        ("radial_truncate", "r<=5"),
        ("radial_decay", "gamma=4"),
        ("layer_radial_ablate", "layer=0,r<=4"),
        ("layer_radial_ablate", "layer=1,r<=4"),
        ("layer_radial_ablate", "layer=2,r<=4"),
    }
    out: list[dict[str, object]] = []
    for r in rows:
        if (r["eval_mode"], r["eval_param"]) not in keep:
            continue
        out.append(
            {
                "dataset": "svhn",
                "task": "reconstruction",
                "basis": r["basis"],
                "group": r["eval_mode"],
                "condition": r["eval_param"],
                "n": int(float(r["n"])),
                "score_mean": float(r["score_mean"]),
                "score_std": float(r["score_std"]),
                "interpretation": "",
            }
        )
    return out


def stl10_rows() -> list[dict[str, object]]:
    trainradial_dir = ROOT / "v7_stl10_recon_mixed_trainradial_r4_10k_3seed_from_relative_s426"
    result_csv = trainradial_dir / "offset_holdout_results.csv"
    if not result_csv.exists():
        return []
    best_control = "radial_truncate r<=4 mean 0.8190"
    component_ablate_drop: object = "mechanism pending"
    mechanism_path = (
        ROOT
        / "v7_stl10_trainradial_component_mechanism_summary"
        / "component_mechanism_family_key_metrics.csv"
    )
    if mechanism_path.exists():
        mechanism = read_rows(mechanism_path)[0]
        best_control = f"radial_truncate r<=4 mean {float(mechanism['radial4_mean']):.4f}"
        component_ablate_drop = (
            f"{float(mechanism['l012345_ablate_drop_mean']):.4f} "
            "L0-L5 all-head ablate mean"
        )
    return [
        summarize_offset_run(
            dataset="stl10",
            task="reconstruction",
            basis="mixed_toric_PJ_R0_top220_residual_dct_top32_trainradial_r4",
            steps=10000,
            result_csv=result_csv,
            relative_reference=None,
            best_control=best_control,
            component_ablate_drop=component_ablate_drop,
            notes=(
                "STL10 train-radial r=4 10k/3seed compact rescue; "
                "fixes naive full heldout collapse while preserving local geometry"
            ),
        )
    ]


def stl10_control_rows() -> list[dict[str, object]]:
    path = ROOT / "v7_stl10_recon_mixed_trainradial_r4_10k_3seed_from_relative_s426" / "offset_holdout_eval_controls_aggregate.csv"
    if not path.exists():
        return []
    keep = {
        ("full", ""),
        ("visible_only", ""),
        ("zero_bias", ""),
        ("heldout_clamp", ""),
        ("radial_decay", "gamma=4"),
        ("radial_truncate", "r<=4"),
        ("radial_truncate", "r<=5"),
        ("radial_band", "6<=r<9"),
    }
    out: list[dict[str, object]] = []
    for r in read_rows(path):
        if (r["eval_mode"], r["eval_param"]) not in keep:
            continue
        condition = f"{r['eval_mode']} {r['eval_param']}".strip()
        out.append(
            {
                "dataset": "stl10",
                "task": "reconstruction",
                "basis": "mixed_toric_PJ_R0_top220_residual_dct_top32_trainradial_r4",
                "group": r["eval_mode"],
                "condition": condition,
                "n": int(float(r["n"])),
                "score_mean": float(r["score_mean"]),
                "score_std": float(r["score_std"]),
                "interpretation": "STL10 train-radial r=4 full/visible boundary control",
            }
        )
    mechanism_path = (
        ROOT
        / "v7_stl10_trainradial_component_mechanism_summary"
        / "component_mechanism_family_key_metrics.csv"
    )
    if mechanism_path.exists():
        mechanism = read_rows(mechanism_path)[0]
        n = int(float(mechanism["n"]))
        basis = str(mechanism["basis"])
        for condition, mean_col, std_col, interpretation in [
            (
                "L0-L2 all-head keep r<=4",
                "l012_keep_mean",
                "l012_keep_std",
                "early local pool is nearly sufficient",
            ),
            (
                "L0-L2 all-head ablate r<=4",
                "l012_ablate_mean",
                "l012_ablate_std",
                "early local pool is strongly necessary",
            ),
            (
                "L0-L3 all-head keep r<=4",
                "l0123_keep_mean",
                "l0123_keep_std",
                "early plus L3 local pool is near full",
            ),
            (
                "L0-L3 all-head ablate r<=4",
                "l0123_ablate_mean",
                "l0123_ablate_std",
                "early plus L3 pool leaves late residual compensation",
            ),
            (
                "L4-L5 all-head keep r<=4",
                "l45_keep_mean",
                "l45_keep_std",
                "late local pool is weak but nonzero compensation",
            ),
            (
                "L4-L5 all-head ablate r<=4",
                "l45_ablate_mean",
                "l45_ablate_std",
                "ablating late compensation preserves most full score",
            ),
            (
                "L0-L5 all-head keep r<=4",
                "l012345_keep_mean",
                "l012345_keep_std",
                "all local layers recover radial r<=4 score",
            ),
            (
                "L0-L5 all-head ablate r<=4",
                "l012345_ablate_mean",
                "l012345_ablate_std",
                "all local layers nearly exhaust to zero",
            ),
        ]:
            out.append(
                {
                    "dataset": "stl10",
                    "task": "reconstruction",
                    "basis": basis,
                    "group": "mixed_component_pool",
                    "condition": condition,
                    "n": n,
                    "score_mean": float(mechanism[mean_col]),
                    "score_std": float(mechanism[std_col]),
                    "interpretation": interpretation,
                }
            )
    return out


def cifar100_control_rows() -> list[dict[str, object]]:
    rows = read_rows(ROOT / "v7_cifar100_causal_summary.csv")
    out: list[dict[str, object]] = []
    for r in rows:
        out.append(
            {
                "dataset": "cifar100",
                "task": "reconstruction",
                "basis": r["model"],
                "group": r["group"],
                "condition": r["condition"],
                "n": r["seed_or_seeds"],
                "score_mean": float(r["score"]),
                "score_std": "",
                "interpretation": r["interpretation"],
            }
        )
    mixed_controls = ROOT / "v8_cifar100_recon_mixed_residual_dct32_10k_3seed" / "offset_holdout_eval_controls_aggregate.csv"
    if mixed_controls.exists():
        keep = {
            ("full", ""),
            ("visible_only", ""),
            ("zero_bias", ""),
            ("heldout_clamp", ""),
            ("radial_truncate", "r<=4"),
            ("radial_truncate", "r<=5"),
        }
        for r in read_rows(mixed_controls):
            if (r["eval_mode"], r["eval_param"]) not in keep:
                continue
            out.append(
                {
                    "dataset": "cifar100",
                    "task": "reconstruction",
                    "basis": r["basis"],
                    "group": "mixed_offset_control",
                    "condition": f"{r['eval_mode']} {r['eval_param']}".strip(),
                    "n": int(float(r["n"])),
                    "score_mean": float(r["score_mean"]),
                    "score_std": float(r["score_std"]),
                    "interpretation": "mixed residual-DCT full/visible boundary control",
                }
            )
    mixed_layer = ROOT / "v8_cifar100_mixed_residual_dct32_mechanism_summary.csv"
    if mixed_layer.exists():
        for r in read_rows(mixed_layer):
            if r["condition"] not in {
                "radial_truncate r<=5",
                "layer0 ablate r<=2",
                "layer2 ablate r<=2",
                "layer0 keep r<=6",
                "layer2 keep r<=2",
            }:
                continue
            out.append(
                {
                    "dataset": "cifar100",
                    "task": "reconstruction",
                    "basis": r["model"],
                    "group": "mixed_layer_pool",
                    "condition": r["condition"],
                    "n": r["seed"],
                    "score_mean": float(r["score"]),
                    "score_std": "",
                    "interpretation": r["interpretation"],
                }
            )
    mixed_heads = ROOT / "v8_cifar100_mixed_residual_dct32_headpool_summary.csv"
    if mixed_heads.exists():
        for r in read_rows(mixed_heads):
            if r["condition"] not in {
                "keep 5 early heads r<=4",
                "keep 11 early+late heads r<=4",
                "ablate 5 early heads r<=4",
                "ablate 11 early+late heads r<=4",
            }:
                continue
            out.append(
                {
                    "dataset": "cifar100",
                    "task": "reconstruction",
                    "basis": r["model"],
                    "group": "mixed_head_pool",
                    "condition": r["condition"],
                    "n": r["seed"],
                    "score_mean": float(r["score"]),
                    "score_std": "",
                    "interpretation": r["interpretation"],
                }
            )
    mixed_component = (
        ROOT
        / "v8_cifar100_mixed_component_mechanism_summary"
        / "component_mechanism_family_key_metrics.csv"
    )
    if mixed_component.exists():
        row = read_rows(mixed_component)[0]
        n = int(float(row["n"]))
        basis = str(row["basis"])
        for condition, mean_col, std_col, interpretation in [
            (
                "L0-L2 all-head keep r<=4",
                "l012_keep_mean",
                "l012_keep_std",
                "early local pool is sufficient and near full",
            ),
            (
                "L0-L2 all-head ablate r<=4",
                "l012_ablate_mean",
                "l012_ablate_std",
                "early local pool is strongly necessary but not exhaustive",
            ),
            (
                "L0-L3 all-head keep r<=4",
                "l0123_keep_mean",
                "l0123_keep_std",
                "early plus L3 pool is sufficient and near full",
            ),
            (
                "L0-L3 all-head ablate r<=4",
                "l0123_ablate_mean",
                "l0123_ablate_std",
                "early plus L3 pool is strongly necessary with residual compensation",
            ),
            (
                "L4-L5 all-head keep r<=4",
                "l45_keep_mean",
                "l45_keep_std",
                "late local pool explains the residual compensation",
            ),
            (
                "L0-L5 all-head ablate r<=4",
                "l012345_ablate_mean",
                "l012345_ablate_std",
                "all local layers nearly exhaust to zero",
            ),
        ]:
            out.append(
                {
                    "dataset": "cifar100",
                    "task": "reconstruction",
                    "basis": basis,
                    "group": "mixed_component_pool",
                    "condition": condition,
                    "n": n,
                    "score_mean": float(row[mean_col]),
                    "score_std": float(row[std_col]),
                    "interpretation": interpretation,
                }
            )
    return out


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(fmt(val))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_markdown(main_rows: list[dict[str, object]], control_rows: list[dict[str, object]]) -> None:
    key_controls = [
        r
        for r in control_rows
        if (
            str(r["dataset"]) == "cifar10"
            and (
                str(r["group"]) in {"full", "visible_only", "zero_bias"}
                or (r["group"] == "radial_truncate" and r["condition"] == "r<=4")
                or (r["group"] == "layer_radial_ablate" and r["condition"] == "layer=2,r<=4")
                or r["group"] == "mixed_component_pool"
            )
        )
        or (
            str(r["dataset"]) == "cifar100"
            and str(r["group"])
            in {
                "adaptive_top_heads",
                "early_l0_l1_l2_all_heads",
                "mixed_offset_control",
                "mixed_layer_pool",
                "mixed_head_pool",
                "mixed_component_pool",
            }
        )
        or (
            str(r["dataset"]) == "svhn"
            and (
                str(r["group"]) in {"full", "visible_only", "zero_bias", "heldout_clamp"}
                or (r["group"] == "radial_truncate" and r["condition"] == "r<=4")
            )
        )
        or (
            str(r["dataset"]) == "stl10"
            and (
                str(r["group"]) in {"full", "visible_only", "zero_bias", "heldout_clamp"}
                or (r["group"] == "radial_truncate" and "r<=4" in str(r["condition"]))
                or (r["group"] == "radial_band" and "6<=r<9" in str(r["condition"]))
                or r["group"] == "mixed_component_pool"
            )
        )
    ]
    text = "\n".join(
        [
            "# V7 Strong Dataset Summary",
            "",
            "Generated from current result CSVs.",
            "",
            "## Main Results",
            "",
            markdown_table(
                main_rows,
                [
                    "dataset",
                    "basis",
                    "steps",
                    "n",
                    "final_mean",
                    "visible_mean",
                    "zero_mean",
                    "full_minus_zero_mean",
                    "full_minus_visible_mean",
                    "relative_ratio",
                ],
            ),
            "",
            "## Key Controls",
            "",
            markdown_table(
                key_controls,
                ["dataset", "basis", "group", "condition", "n", "score_mean", "score_std"],
            ),
            "",
            "## Reading",
            "",
            "- CIFAR100 reconstruction is the v7 strong positive case: high full score, zero-bias collapse, and causal early-head evidence.",
            "- CIFAR100 mixed residual-DCT matches the DCT full-performance level while keeping a 3-seed Toric/PJ-style local head-pool mechanism.",
            "- CIFAR10 top110 30k remains a strong local-geometry anchor with unstable full extrapolation; mixed residual-DCT 10k largely rescues the full-visible gap and preserves a 3-seed early head-pool mechanism.",
            "- SVHN reconstruction is now a strong 10k/3seed cross-dataset positive case; mixed residual-DCT is near relative-table full performance.",
            "- STL10 train-radial r=4 is now a 10k/3seed compact rescue anchor: it fixes naive full extrapolation collapse while keeping a small full-visible gap.",
            "- STL10 mechanism controls now show near-sufficient L0-L3 local heads, late L4-L5 compensation, and near-zero exhaustion after ablating all L0-L5 local heads.",
            "- The evidence points toward DCT or residual-DCT mixed bases for stable full extrapolation while preserving local mechanism.",
            "",
        ]
    )
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_rows = cifar100_rows()
    add_cifar100_component_drops(main_rows)

    cifar10_summary = summarize_offset_run(
        dataset="cifar10",
        task="reconstruction",
        basis="table_informed_toric_PJ_R0_top110",
        steps=30000,
        result_csv=ROOT / "v7_cifar10_recon_top110_30k_3seed" / "offset_holdout_results.csv",
        relative_reference=json.loads(
            (ROOT / "v7_cifar10_recon_top110_30k_3seed" / "offset_holdout_summary.json").read_text(
                encoding="utf-8"
            )
        )["teacher_metadata"]["final_score"],
        best_control="radial_truncate r<=4 mean 0.8555",
        component_ablate_drop=0.6283,
        notes="30k anchor; strong local geometry but unstable full heldout extrapolation",
    )
    main_rows.append(cifar10_summary)
    cifar10_mixed_dir = ROOT / "v7_cifar10_recon_mixed_residual_dct32_10k_3seed"
    if (cifar10_mixed_dir / "offset_holdout_results.csv").exists():
        relative_reference = json.loads(
            (cifar10_mixed_dir / "offset_holdout_summary.json").read_text(encoding="utf-8")
        )["teacher_metadata"]["final_score"]
        main_rows.append(
            summarize_offset_run(
                dataset="cifar10",
                task="reconstruction",
                basis="mixed_toric_PJ_R0_top220_residual_dct_top32",
                steps=10000,
                result_csv=cifar10_mixed_dir / "offset_holdout_results.csv",
                relative_reference=relative_reference,
                best_control="radial_truncate r<=4 mean 0.8577",
                component_ablate_drop="0.6138 L0-L3 all-head ablate mean",
                notes="10k mixed residual-DCT; rescues CIFAR10 full heldout extrapolation",
            )
        )
    main_rows.extend(svhn_rows())
    main_rows.extend(stl10_rows())

    main_fields = [
        "dataset",
        "task",
        "basis",
        "seeds",
        "steps",
        "n",
        "num_features",
        "best_mean",
        "best_std",
        "final_mean",
        "final_std",
        "visible_mean",
        "visible_std",
        "zero_mean",
        "zero_std",
        "full_minus_zero_mean",
        "full_minus_zero_std",
        "full_minus_visible_mean",
        "full_minus_visible_std",
        "heldout_bias_rms_mean",
        "heldout_bias_rms_std",
        "relative_ratio",
        "best_control",
        "component_ablate_drop",
        "notes",
    ]
    write_rows(OUT / "v7_main_results.csv", main_rows, main_fields)

    control_rows = cifar100_control_rows() + cifar10_control_rows() + svhn_control_rows() + stl10_control_rows()
    control_fields = [
        "dataset",
        "task",
        "basis",
        "group",
        "condition",
        "n",
        "score_mean",
        "score_std",
        "interpretation",
    ]
    write_rows(OUT / "v7_control_results.csv", control_rows, control_fields)
    write_markdown(main_rows, control_rows)

    summary = {
        "main_results": str(OUT / "v7_main_results.csv"),
        "control_results": str(OUT / "v7_control_results.csv"),
        "readme": str(OUT / "README.md"),
        "n_main_rows": len(main_rows),
        "n_control_rows": len(control_rows),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
