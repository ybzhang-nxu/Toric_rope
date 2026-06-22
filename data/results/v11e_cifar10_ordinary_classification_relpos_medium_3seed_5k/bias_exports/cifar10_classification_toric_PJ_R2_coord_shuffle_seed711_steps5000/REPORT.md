# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: classification
- basis: toric_PJ_R2_coord_shuffle
- seed: 711
- steps: 5000
- best_step: 2000
- score: 0.5694000124931335
- final_score: 0.5685999989509583
- train_score: 1.0
- grid_side: 8
- n_layers: 4
- n_heads: 8
- num_features: 55

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| toric_PJ_R2_coord_shuffle | centered | interior_only | 32 | 0.8634 | 0.0152 | 0.9226 | 0.1033 | 0.1612 |
| toric_PJ_R2_coord_shuffle | centered | reflect_padding | 32 | 0.8634 | 0.0102 | 0.9226 | 0.1033 | 0.1612 |
| toric_PJ_R2_coord_shuffle | normalized | interior_only | 32 | 0.8634 | 0.0152 | 0.9226 | 0.1033 | 0.1612 |
| toric_PJ_R2_coord_shuffle | normalized | reflect_padding | 32 | 0.8634 | 0.0102 | 0.9226 | 0.1033 | 0.1612 |
| toric_PJ_R2_coord_shuffle | raw | interior_only | 32 | 0.8440 | 0.0152 | 0.9226 | 0.1033 | 0.1612 |
| toric_PJ_R2_coord_shuffle | raw | reflect_padding | 32 | 0.8440 | 0.0102 | 0.9226 | 0.1033 | 0.1612 |

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
