# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: classification
- basis: no_pos_constant
- seed: 526
- steps: 5000
- best_step: 2000
- score: 0.5679999589920044
- final_score: 0.5654000043869019
- train_score: 1.0
- grid_side: 8
- n_layers: 4
- n_heads: 8
- num_features: 1

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| no_pos_constant | centered | interior_only | 32 | 0.0000 | 0.0000 | 0.0014 | 0.0004 | 0.0000 |
| no_pos_constant | centered | reflect_padding | 32 | 0.0000 | 0.0000 | 0.0014 | 0.0004 | 0.0000 |
| no_pos_constant | normalized | interior_only | 32 | 0.4367 | 0.0628 | 0.5776 | 0.4703 | 0.7506 |
| no_pos_constant | normalized | reflect_padding | 32 | 0.4367 | 0.0520 | 0.5776 | 0.4703 | 0.7506 |
| no_pos_constant | raw | interior_only | 32 | 0.0000 | 0.0000 | 0.0014 | 0.0004 | 0.0000 |
| no_pos_constant | raw | reflect_padding | 32 | 0.0000 | 0.0000 | 0.0014 | 0.0004 | 0.0000 |

Notes:

- Geometry is reported for raw, centered, and normalized gauges.
- Main-text curvature should use `interior_only`; `reflect_padding` is a sanity check.
- `obl_ratio` is the energy share outside the best full axial-additive projection.

Artifacts:

- `bias_tables.npz`
- `geometry_metrics.csv`
- `geometry_aggregate.csv`
- `spectral_peaks.csv`
- `mixed_hessian_heatmaps.png`
- `fft_spectrum_grid.png`
- `dct_spectrum_grid.png`
- `axial_residual_heatmaps.png`
