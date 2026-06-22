# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: classification
- basis: toric_PJ_R2_coord_shuffle
- seed: 511
- steps: 5000
- best_step: 4500
- score: 0.5823999643325806
- final_score: 0.5807999968528748
- train_score: 1.0
- grid_side: 8
- n_layers: 4
- n_heads: 8
- num_features: 55

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| toric_PJ_R2_coord_shuffle | centered | interior_only | 32 | 0.8793 | 0.0151 | 0.9277 | 0.0936 | 0.1600 |
| toric_PJ_R2_coord_shuffle | centered | reflect_padding | 32 | 0.8793 | 0.0100 | 0.9277 | 0.0936 | 0.1600 |
| toric_PJ_R2_coord_shuffle | normalized | interior_only | 32 | 0.8793 | 0.0151 | 0.9277 | 0.0936 | 0.1600 |
| toric_PJ_R2_coord_shuffle | normalized | reflect_padding | 32 | 0.8793 | 0.0100 | 0.9277 | 0.0936 | 0.1600 |
| toric_PJ_R2_coord_shuffle | raw | interior_only | 32 | 0.8674 | 0.0151 | 0.9277 | 0.0936 | 0.1600 |
| toric_PJ_R2_coord_shuffle | raw | reflect_padding | 32 | 0.8674 | 0.0100 | 0.9277 | 0.0936 | 0.1600 |

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
