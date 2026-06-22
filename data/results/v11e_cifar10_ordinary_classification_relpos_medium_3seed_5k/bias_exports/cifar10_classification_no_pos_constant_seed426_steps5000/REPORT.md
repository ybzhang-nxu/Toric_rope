# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: classification
- basis: no_pos_constant
- seed: 426
- steps: 5000
- best_step: 4999
- score: 0.5708000063896179
- final_score: 0.5708000063896179
- train_score: 1.0
- grid_side: 8
- n_layers: 4
- n_heads: 8
- num_features: 1

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| no_pos_constant | centered | interior_only | 32 | 0.0000 | 0.0000 | 0.0010 | 0.0003 | 0.0000 |
| no_pos_constant | centered | reflect_padding | 32 | 0.0000 | 0.0000 | 0.0010 | 0.0003 | 0.0000 |
| no_pos_constant | normalized | interior_only | 32 | 0.3357 | 0.0484 | 0.5758 | 0.4602 | 0.6936 |
| no_pos_constant | normalized | reflect_padding | 32 | 0.3357 | 0.0399 | 0.5758 | 0.4602 | 0.6936 |
| no_pos_constant | raw | interior_only | 32 | 0.0000 | 0.0000 | 0.0010 | 0.0003 | 0.0000 |
| no_pos_constant | raw | reflect_padding | 32 | 0.0000 | 0.0000 | 0.0010 | 0.0003 | 0.0000 |

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
