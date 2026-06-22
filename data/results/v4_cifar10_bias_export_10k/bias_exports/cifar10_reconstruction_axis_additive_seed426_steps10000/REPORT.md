# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: axis_additive
- seed: 426
- steps: 10000
- best_step: 9999
- score: 0.6042166948318481
- final_score: 0.6034532785415649
- train_score: 0.609957218170166
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 5

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| axis_additive | centered | interior_only | 48 | 0.0000 | 0.0000 | 0.3493 | 0.8757 | 0.9315 |
| axis_additive | centered | reflect_padding | 48 | 0.0000 | 0.0000 | 0.3493 | 0.8757 | 0.9315 |
| axis_additive | normalized | interior_only | 48 | 0.0000 | 0.0000 | 0.3493 | 0.8757 | 0.9315 |
| axis_additive | normalized | reflect_padding | 48 | 0.0000 | 0.0000 | 0.3493 | 0.8757 | 0.9315 |
| axis_additive | raw | interior_only | 48 | 0.0000 | 0.0000 | 0.3493 | 0.8757 | 0.9315 |
| axis_additive | raw | reflect_padding | 48 | 0.0000 | 0.0000 | 0.3493 | 0.8757 | 0.9315 |

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
