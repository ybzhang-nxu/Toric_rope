# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: toric_PJ_R2_coord_shuffle
- seed: 494
- steps: 10000
- best_step: 9999
- score: 0.39266371726989746
- final_score: 0.39192378520965576
- train_score: 0.4086872339248657
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 55

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| toric_PJ_R2_coord_shuffle | centered | interior_only | 48 | 0.8486 | 0.0150 | 0.9241 | 0.1035 | 0.1542 |
| toric_PJ_R2_coord_shuffle | centered | reflect_padding | 48 | 0.8486 | 0.0094 | 0.9241 | 0.1035 | 0.1542 |
| toric_PJ_R2_coord_shuffle | normalized | interior_only | 48 | 0.8486 | 0.0150 | 0.9241 | 0.1035 | 0.1542 |
| toric_PJ_R2_coord_shuffle | normalized | reflect_padding | 48 | 0.8486 | 0.0094 | 0.9241 | 0.1035 | 0.1542 |
| toric_PJ_R2_coord_shuffle | raw | interior_only | 48 | 0.8357 | 0.0150 | 0.9241 | 0.1035 | 0.1542 |
| toric_PJ_R2_coord_shuffle | raw | reflect_padding | 48 | 0.8357 | 0.0094 | 0.9241 | 0.1035 | 0.1542 |

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
