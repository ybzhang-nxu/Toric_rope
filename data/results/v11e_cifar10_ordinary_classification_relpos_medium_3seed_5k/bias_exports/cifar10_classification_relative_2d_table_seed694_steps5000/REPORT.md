# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: classification
- basis: relative_2d_table
- seed: 694
- steps: 5000
- best_step: 4999
- score: 0.5663999915122986
- final_score: 0.5663999915122986
- train_score: 1.0
- grid_side: 8
- n_layers: 4
- n_heads: 8
- num_features: 225

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| relative_2d_table | centered | interior_only | 32 | 0.4152 | 0.0194 | 0.7117 | 0.4478 | 0.5609 |
| relative_2d_table | centered | reflect_padding | 32 | 0.4152 | 0.0107 | 0.7117 | 0.4478 | 0.5609 |
| relative_2d_table | normalized | interior_only | 32 | 0.4152 | 0.0194 | 0.7117 | 0.4478 | 0.5609 |
| relative_2d_table | normalized | reflect_padding | 32 | 0.4152 | 0.0107 | 0.7117 | 0.4478 | 0.5609 |
| relative_2d_table | raw | interior_only | 32 | 0.3757 | 0.0194 | 0.7117 | 0.4478 | 0.5609 |
| relative_2d_table | raw | reflect_padding | 32 | 0.3757 | 0.0107 | 0.7117 | 0.4478 | 0.5609 |

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
