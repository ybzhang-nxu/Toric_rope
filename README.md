# Toric PJ Two-Dimensional Relative Attention: Reproducibility Bundle

This bundle is the artifact package for the current paper draft,
**Toric PJ Position Spaces for Scalar Two-Dimensional Relative Attention**.
It is intended to be uploaded directly as a public GitHub repository root, or
kept as `MetricToric/reproducibility/paper_bundle/` inside the larger working
tree.  The artifact IDs here match the paper's Artifact Index.

The bundle is intentionally scoped to experiments that appear in the current
paper or its appendices.  It copies the code snapshot, paper figure outputs,
and derived data/result records needed to audit the reported figures and
tables.  It does not include raw CIFAR10, TinyImageNet, or NSynth audio
datasets, full training logs, or model checkpoints.  The included NSynth CQT
arrays are derived feature/cache records used by the reported downstream
boundary audit, not raw audio.

## Layout

- `ARTIFACT_MANIFEST.tsv`: stable `MT-Axx` artifact IDs and their paper use.
- `CODE_MANIFEST.tsv`: experiment entrypoints and implementation files.
- `DATA_RESULT_MANIFEST.tsv`: copied result records and compact/source-only
  notes.
- `FIGURE_MANIFEST.tsv`: current paper figures and figure source packages
  included in this bundle.
- `LICENSE`: MIT license for this reproducibility bundle.
- `code/`: copied source snapshot for paper-relevant runners, implementations,
  models, diagnostics, and figure builders.
- `data/results/`: copied result directories, preserving the original
  `MetricToric/results/...` names.
- `figures/`: copied figure PDFs and source packages used by the paper.
- `environment/requirements-v11.txt`: training/export dependency snapshot.
- `scripts/verify_paper_bundle.py`: stdlib-only manifest checker.
- `.gitignore`: local cache/environment ignores suitable for a GitHub upload.

## Quick Checks

From this directory:

```bash
python3 scripts/verify_paper_bundle.py
```

To rebuild the paper figures from the full repository layout:

```bash
cd ../../paper
python3 scripts/build_paper_figures.py
python3 scripts/build_revision_figures.py
```

The copied figure builders under `code/paper_scripts/` are provenance snapshots.
The commands above use the live repository paths so that the generated PDFs land
in `MetricToric/paper/figures/`.

## License

This reproducibility bundle is released under the MIT license; see `LICENSE`.
Third-party raw datasets are not included here and remain governed by their own
dataset terms.  The included result tables, figures, and derived feature/cache
records are provided as part of this paper artifact package.

## GitHub Release Checklist

Suggested pre-upload check:

```bash
python3 scripts/verify_paper_bundle.py
```

The checker validates manifest paths, artifact IDs, registered figure files, and
common absolute-path leaks.

## Scope Notes

Most copied records are direct table/figure inputs.  Two appendix/control
artifacts are compacted to avoid turning this bundle into a full raw-sweep
archive:

- `MT-A10` controlled recovery: copied aggregate, report, summary, and
  diagnostic PDFs; omitted raw sweep rows and raw ridge paths.
- `MT-A10` conditioning audit: copied aggregate, report, summary, bootstrap
  stability, and diagnostic PDFs; omitted raw sweep rows and raw ridge paths.

The omitted raw sweep files remain in the original repository results directory
and are recorded as `source_only` rows in `DATA_RESULT_MANIFEST.tsv`.
