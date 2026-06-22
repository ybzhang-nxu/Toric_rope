#!/usr/bin/env python3
"""Check the current-paper reproducibility bundle manifests."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
ID_RE = re.compile(r"^MT-A\d{2}$")
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_TEXT_PATTERNS = (
    "/" + "home" + "/",
    "." + "codex",
    "chat" + "gpt",
    "utm_" + "source",
)


def read_tsv(name: str) -> list[dict[str, str]]:
    path = BUNDLE_ROOT / name
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def check_bundle_path(
    errors: list[str],
    manifest: str,
    row_number: int,
    row: dict[str, str],
    *,
    allow_source_only: bool = False,
) -> None:
    bundle_path = row.get("bundle_path", "").strip()
    status = row.get("copy_status", "").strip()
    if allow_source_only and status == "source_only":
        return
    if not bundle_path:
        errors.append(f"{manifest}:{row_number}: missing bundle_path")
        return
    if not (BUNDLE_ROOT / bundle_path).exists():
        errors.append(f"{manifest}:{row_number}: missing {bundle_path}")


def listed_bundle_paths(manifest: str, bundle_path_column: str = "bundle_path") -> set[Path]:
    paths: set[Path] = set()
    for row in read_tsv(manifest):
        bundle_path = row.get(bundle_path_column, "").strip()
        if bundle_path:
            paths.add(Path(bundle_path))
    return paths


def check_registered_figures(errors: list[str]) -> None:
    figure_dir = BUNDLE_ROOT / "figures"
    listed = listed_bundle_paths("FIGURE_MANIFEST.tsv")
    actual = {
        path.relative_to(BUNDLE_ROOT)
        for path in figure_dir.iterdir()
        if path.is_file()
    }
    extra = sorted(actual - listed)
    missing = sorted(listed - actual)
    for path in extra:
        errors.append(f"figures: unregistered file {path}")
    for path in missing:
        errors.append(f"FIGURE_MANIFEST.tsv: listed figure missing {path}")


def check_public_text_hygiene(errors: list[str]) -> None:
    for path in BUNDLE_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel_path = path.relative_to(BUNDLE_ROOT)
        if rel_path == Path("scripts/verify_paper_bundle.py"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"{rel_path}: text-like suffix is not UTF-8")
            continue
        lowered = text.lower()
        for pattern in FORBIDDEN_TEXT_PATTERNS:
            if pattern in lowered:
                errors.append(f"{rel_path}: forbidden public-release pattern {pattern!r}")


def main() -> int:
    errors: list[str] = []
    artifact_rows = read_tsv("ARTIFACT_MANIFEST.tsv")
    artifact_ids = [row["artifact_id"].strip() for row in artifact_rows]
    artifact_id_set = set(artifact_ids)

    if len(artifact_ids) != len(artifact_id_set):
        errors.append("ARTIFACT_MANIFEST.tsv: duplicate artifact IDs")
    for row_number, artifact_id in enumerate(artifact_ids, start=2):
        if not ID_RE.match(artifact_id):
            errors.append(
                f"ARTIFACT_MANIFEST.tsv:{row_number}: bad artifact_id {artifact_id!r}"
            )

    for manifest in ("CODE_MANIFEST.tsv", "FIGURE_MANIFEST.tsv"):
        rows = read_tsv(manifest)
        for row_number, row in enumerate(rows, start=2):
            artifact_id = row.get("artifact_id", "").strip()
            if artifact_id not in artifact_id_set:
                errors.append(f"{manifest}:{row_number}: unknown {artifact_id!r}")
            check_bundle_path(errors, manifest, row_number, row)

    rows = read_tsv("DATA_RESULT_MANIFEST.tsv")
    for row_number, row in enumerate(rows, start=2):
        artifact_id = row.get("artifact_id", "").strip()
        if artifact_id not in artifact_id_set:
            errors.append(f"DATA_RESULT_MANIFEST.tsv:{row_number}: unknown {artifact_id!r}")
        status = row.get("copy_status", "").strip()
        if status not in {"copied", "compact", "source_only"}:
            errors.append(
                f"DATA_RESULT_MANIFEST.tsv:{row_number}: bad copy_status {status!r}"
            )
        check_bundle_path(
            errors,
            "DATA_RESULT_MANIFEST.tsv",
            row_number,
            row,
            allow_source_only=True,
        )

    check_registered_figures(errors)
    check_public_text_hygiene(errors)

    if errors:
        print("Bundle verification failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    copied_results = sum(
        1
        for row in rows
        if row.get("copy_status", "").strip() in {"copied", "compact"}
    )
    print("Bundle verification passed.")
    print(f"  artifacts: {len(artifact_rows)}")
    print(f"  code records: {len(read_tsv('CODE_MANIFEST.tsv'))}")
    print(f"  figure records: {len(read_tsv('FIGURE_MANIFEST.tsv'))}")
    print(f"  copied/compact result records: {copied_results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
