# V4 Relative Table Geometry Report

Metadata:

- inputs: 5
- device: cpu
- topk_metrics: 5
- topk_peaks: 8
- first_dataset: cifar10
- first_task: reconstruction
- first_basis: axis_additive
- first_seed: 426
- first_steps: 10000
- first_best_step: 9999
- first_score: 0.6042166948318481
- first_final_score: 0.6034532785415649
- first_train_score: 0.609957218170166
- first_grid_side: 8
- first_n_layers: 6
- first_n_heads: 8
- first_num_features: 5
- first_bias_npz: results/v4_cifar10_bias_export_10k/bias_exports/cifar10_reconstruction_axis_additive_seed426_steps10000/bias_tables.npz

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| axis_additive | centered | interior_only | 48 | 0.0000 | 0.0000 | 0.3493 | 0.8757 | 0.9315 |
| axis_additive | centered | reflect_padding | 48 | 0.0000 | 0.0000 | 0.3493 | 0.8757 | 0.9315 |
| axis_additive | normalized | interior_only | 48 | 0.0000 | 0.0000 | 0.3493 | 0.8757 | 0.9315 |
| axis_additive | normalized | reflect_padding | 48 | 0.0000 | 0.0000 | 0.3493 | 0.8757 | 0.9315 |
| axis_additive | raw | interior_only | 48 | 0.0000 | 0.0000 | 0.3493 | 0.8757 | 0.9315 |
| axis_additive | raw | reflect_padding | 48 | 0.0000 | 0.0000 | 0.3493 | 0.8757 | 0.9315 |
| pruned_toric_PJ | centered | interior_only | 48 | 0.7933 | 0.1217 | 0.6408 | 0.4904 | 0.5894 |
| pruned_toric_PJ | centered | reflect_padding | 48 | 0.7933 | 0.0394 | 0.6408 | 0.4904 | 0.5894 |
| pruned_toric_PJ | normalized | interior_only | 48 | 0.7933 | 0.1217 | 0.6408 | 0.4904 | 0.5894 |
| pruned_toric_PJ | normalized | reflect_padding | 48 | 0.7933 | 0.0394 | 0.6408 | 0.4904 | 0.5894 |
| pruned_toric_PJ | raw | interior_only | 48 | 0.7823 | 0.1217 | 0.6408 | 0.4904 | 0.5894 |
| pruned_toric_PJ | raw | reflect_padding | 48 | 0.7823 | 0.0394 | 0.6408 | 0.4904 | 0.5894 |
| relative_2d_table | centered | interior_only | 48 | 0.2334 | 0.0193 | 0.5913 | 0.5705 | 0.6955 |
| relative_2d_table | centered | reflect_padding | 48 | 0.2334 | 0.0136 | 0.5913 | 0.5705 | 0.6955 |
| relative_2d_table | normalized | interior_only | 48 | 0.2334 | 0.0193 | 0.5913 | 0.5705 | 0.6955 |
| relative_2d_table | normalized | reflect_padding | 48 | 0.2334 | 0.0136 | 0.5913 | 0.5705 | 0.6955 |
| relative_2d_table | raw | interior_only | 48 | 0.1447 | 0.0193 | 0.5913 | 0.5705 | 0.6955 |
| relative_2d_table | raw | reflect_padding | 48 | 0.1447 | 0.0136 | 0.5913 | 0.5705 | 0.6955 |
| toric_PJ_R2 | centered | interior_only | 48 | 0.7500 | 0.1290 | 0.6857 | 0.3880 | 0.5618 |
| toric_PJ_R2 | centered | reflect_padding | 48 | 0.7500 | 0.0348 | 0.6857 | 0.3880 | 0.5618 |
| toric_PJ_R2 | normalized | interior_only | 48 | 0.7500 | 0.1290 | 0.6857 | 0.3880 | 0.5618 |
| toric_PJ_R2 | normalized | reflect_padding | 48 | 0.7500 | 0.0348 | 0.6857 | 0.3880 | 0.5618 |
| toric_PJ_R2 | raw | interior_only | 48 | 0.7410 | 0.1290 | 0.6857 | 0.3880 | 0.5618 |
| toric_PJ_R2 | raw | reflect_padding | 48 | 0.7410 | 0.0348 | 0.6857 | 0.3880 | 0.5618 |
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
