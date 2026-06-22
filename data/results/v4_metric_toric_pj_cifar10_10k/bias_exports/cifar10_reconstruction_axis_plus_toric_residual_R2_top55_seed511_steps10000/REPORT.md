# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: axis_plus_toric_residual_R2_top55
- seed: 511
- steps: 10000
- best_step: 9999
- score: 0.8229154348373413
- final_score: 0.8224785327911377
- train_score: 0.829456090927124
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 47

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| axis_plus_toric_residual_R2_top55 | centered | interior_only | 48 | 0.7932 | 0.1267 | 0.5573 | 0.5639 | 0.7023 |
| axis_plus_toric_residual_R2_top55 | centered | reflect_padding | 48 | 0.7932 | 0.0474 | 0.5573 | 0.5639 | 0.7023 |
| axis_plus_toric_residual_R2_top55 | normalized | interior_only | 48 | 0.7932 | 0.1267 | 0.5573 | 0.5639 | 0.7023 |
| axis_plus_toric_residual_R2_top55 | normalized | reflect_padding | 48 | 0.7932 | 0.0474 | 0.5573 | 0.5639 | 0.7023 |
| axis_plus_toric_residual_R2_top55 | raw | interior_only | 48 | 0.7928 | 0.1267 | 0.5573 | 0.5639 | 0.7023 |
| axis_plus_toric_residual_R2_top55 | raw | reflect_padding | 48 | 0.7928 | 0.0474 | 0.5573 | 0.5639 | 0.7023 |

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
