# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: axis_plus_toric_residual_R0_top55
- seed: 694
- steps: 10000
- best_step: 9999
- score: 0.8558091521263123
- final_score: 0.8547145128250122
- train_score: 0.8562932014465332
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 55

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| axis_plus_toric_residual_R0_top55 | centered | interior_only | 48 | 0.4977 | 0.0871 | 0.5691 | 0.4365 | 0.6507 |
| axis_plus_toric_residual_R0_top55 | centered | reflect_padding | 48 | 0.4977 | 0.0579 | 0.5691 | 0.4365 | 0.6507 |
| axis_plus_toric_residual_R0_top55 | normalized | interior_only | 48 | 0.4977 | 0.0871 | 0.5691 | 0.4365 | 0.6507 |
| axis_plus_toric_residual_R0_top55 | normalized | reflect_padding | 48 | 0.4977 | 0.0579 | 0.5691 | 0.4365 | 0.6507 |
| axis_plus_toric_residual_R0_top55 | raw | interior_only | 48 | 0.3667 | 0.0871 | 0.5691 | 0.4365 | 0.6507 |
| axis_plus_toric_residual_R0_top55 | raw | reflect_padding | 48 | 0.3667 | 0.0579 | 0.5691 | 0.4365 | 0.6507 |

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
