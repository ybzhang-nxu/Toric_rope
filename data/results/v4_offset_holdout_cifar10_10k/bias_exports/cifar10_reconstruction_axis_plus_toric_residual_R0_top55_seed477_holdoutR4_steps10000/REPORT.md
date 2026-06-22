# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: axis_plus_toric_residual_R0_top55
- seed: 477
- steps: 10000
- best_step: 9999
- score: 0.647406280040741
- final_score: 0.6475540399551392
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 55
- holdout_radius: 4

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| axis_plus_toric_residual_R0_top55 | centered | interior_only | 48 | 0.5732 | 0.0340 | 0.5614 | 0.4609 | 0.6949 |
| axis_plus_toric_residual_R0_top55 | centered | reflect_padding | 48 | 0.5732 | 0.0283 | 0.5614 | 0.4609 | 0.6949 |
| axis_plus_toric_residual_R0_top55 | normalized | interior_only | 48 | 0.5732 | 0.0340 | 0.5614 | 0.4609 | 0.6949 |
| axis_plus_toric_residual_R0_top55 | normalized | reflect_padding | 48 | 0.5732 | 0.0283 | 0.5614 | 0.4609 | 0.6949 |
| axis_plus_toric_residual_R0_top55 | raw | interior_only | 48 | 0.5413 | 0.0340 | 0.5614 | 0.4609 | 0.6949 |
| axis_plus_toric_residual_R0_top55 | raw | reflect_padding | 48 | 0.5413 | 0.0283 | 0.5614 | 0.4609 | 0.6949 |

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
