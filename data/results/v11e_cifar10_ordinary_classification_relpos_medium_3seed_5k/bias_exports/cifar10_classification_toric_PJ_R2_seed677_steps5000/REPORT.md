# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: classification
- basis: toric_PJ_R2
- seed: 677
- steps: 5000
- best_step: 4000
- score: 0.5661999583244324
- final_score: 0.5655999779701233
- train_score: 1.0
- grid_side: 8
- n_layers: 4
- n_heads: 8
- num_features: 55

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| toric_PJ_R2 | centered | interior_only | 32 | 0.8807 | 0.1337 | 0.6167 | 0.5082 | 0.6833 |
| toric_PJ_R2 | centered | reflect_padding | 32 | 0.8807 | 0.0402 | 0.6167 | 0.5082 | 0.6833 |
| toric_PJ_R2 | normalized | interior_only | 32 | 0.8807 | 0.1337 | 0.6167 | 0.5082 | 0.6833 |
| toric_PJ_R2 | normalized | reflect_padding | 32 | 0.8807 | 0.0402 | 0.6167 | 0.5082 | 0.6833 |
| toric_PJ_R2 | raw | interior_only | 32 | 0.8675 | 0.1337 | 0.6167 | 0.5082 | 0.6833 |
| toric_PJ_R2 | raw | reflect_padding | 32 | 0.8675 | 0.0402 | 0.6167 | 0.5082 | 0.6833 |

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
