#!/usr/bin/env python3
"""Export relative-bias geometry artifacts from V4/V6 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import torch

from toric_pj.diagnostics.relative_table_geometry import write_geometry_artifacts


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def coeff_from_state(state: dict[str, torch.Tensor], *, n_layers: int) -> torch.Tensor:
    coeffs: list[torch.Tensor] = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.coeff"
        if key not in state:
            raise KeyError(f"checkpoint state missing {key}")
        coeffs.append(state[key].detach().float().cpu())
    return torch.stack(coeffs, dim=0)


def checkpoint_paths(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.pt")))
        else:
            paths.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    if not unique:
        raise ValueError("no checkpoint inputs found")
    return unique


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_checkpoint(path: Path, *, state_name: str, output_root: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    state_key = f"{state_name}_model_state_dict"
    if state_key not in checkpoint or checkpoint[state_key] is None:
        raise ValueError(f"{path} does not contain {state_key}")
    metadata = dict(checkpoint["metadata"])
    n_layers = int(metadata.get("n_layers", 0))
    if n_layers <= 0:
        raise ValueError(f"{path} metadata has invalid n_layers={n_layers}")
    grid_side = int(metadata.get("grid_side", 0))
    if grid_side <= 0:
        n_positions = int(round(math.sqrt(float(checkpoint["basis_matrix"].shape[0]))))
        grid_side = int(round(math.sqrt(float(n_positions))))
    state = checkpoint[state_key]
    coeff = coeff_from_state(state, n_layers=n_layers)
    basis_matrix = checkpoint["basis_matrix"].detach().float().cpu()
    export_metadata = {
        **metadata,
        "checkpoint": str(path),
        "checkpoint_state": state_name,
        "source_basis_name": checkpoint.get("basis_name", metadata.get("basis", "")),
    }
    export_dir = output_root / f"{path.stem}_{state_name}"
    artifacts = write_geometry_artifacts(
        export_dir,
        basis_matrix=basis_matrix,
        coeff=coeff,
        side=grid_side,
        metadata=export_metadata,
    )
    summary = {
        "checkpoint": str(path),
        "state": state_name,
        "export_dir": str(export_dir),
        "dataset": metadata.get("dataset", ""),
        "task": metadata.get("task", ""),
        "basis": metadata.get("basis", ""),
        "seed": metadata.get("seed", ""),
        "score": metadata.get("score", ""),
        "final_score": metadata.get("final_score", ""),
        **artifacts,
    }
    (export_dir / "checkpoint_export_metadata.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export bias tables from saved V4/V6 checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--states", default="final", help="Comma-separated checkpoint states: final,best")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    rows: list[dict[str, object]] = []
    for checkpoint in checkpoint_paths(args.checkpoints):
        for state in parse_list(args.states):
            rows.append(export_checkpoint(checkpoint, state_name=state, output_root=output_root))
    write_csv(output_root / "exports.csv", rows)
    print(json.dumps({"output_dir": str(output_root), "exports": rows}, indent=2))


if __name__ == "__main__":
    main()
