# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: relative_2d_table
- seed: 477
- steps: 10000
- best_step: 9000
- score: 0.8361638188362122
- final_score: 0.8376511335372925
- train_score: 0.8367145657539368
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 225

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| relative_2d_table | centered | interior_only | 48 | 0.2334 | 0.0193 | 0.5913 | 0.5705 | 0.6955 |
| relative_2d_table | centered | reflect_padding | 48 | 0.2334 | 0.0136 | 0.5913 | 0.5705 | 0.6955 |
| relative_2d_table | normalized | interior_only | 48 | 0.2334 | 0.0193 | 0.5913 | 0.5705 | 0.6955 |
| relative_2d_table | normalized | reflect_padding | 48 | 0.2334 | 0.0136 | 0.5913 | 0.5705 | 0.6955 |
| relative_2d_table | raw | interior_only | 48 | 0.1447 | 0.0193 | 0.5913 | 0.5705 | 0.6955 |
| relative_2d_table | raw | reflect_padding | 48 | 0.1447 | 0.0136 | 0.5913 | 0.5705 | 0.6955 |

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
