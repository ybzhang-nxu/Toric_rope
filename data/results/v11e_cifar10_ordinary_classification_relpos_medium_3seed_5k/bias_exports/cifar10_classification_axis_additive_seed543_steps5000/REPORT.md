# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: classification
- basis: axis_additive
- seed: 543
- steps: 5000
- best_step: 4000
- score: 0.5867999792098999
- final_score: 0.5861999988555908
- train_score: 1.0
- grid_side: 8
- n_layers: 4
- n_heads: 8
- num_features: 5

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| axis_additive | centered | interior_only | 32 | 0.0000 | 0.0000 | 0.3293 | 0.8884 | 0.9531 |
| axis_additive | centered | reflect_padding | 32 | 0.0000 | 0.0000 | 0.3293 | 0.8884 | 0.9531 |
| axis_additive | normalized | interior_only | 32 | 0.0000 | 0.0000 | 0.3293 | 0.8884 | 0.9531 |
| axis_additive | normalized | reflect_padding | 32 | 0.0000 | 0.0000 | 0.3293 | 0.8884 | 0.9531 |
| axis_additive | raw | interior_only | 32 | 0.0000 | 0.0000 | 0.3293 | 0.8884 | 0.9531 |
| axis_additive | raw | reflect_padding | 32 | 0.0000 | 0.0000 | 0.3293 | 0.8884 | 0.9531 |

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
