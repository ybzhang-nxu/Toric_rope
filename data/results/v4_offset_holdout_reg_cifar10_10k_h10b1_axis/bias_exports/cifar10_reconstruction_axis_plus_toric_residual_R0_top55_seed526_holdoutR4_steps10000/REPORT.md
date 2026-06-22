# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: axis_plus_toric_residual_R0_top55
- seed: 526
- steps: 10000
- best_step: 9000
- score: 0.7339975833892822
- final_score: 0.7327132225036621
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 55
- holdout_radius: 4
- heldout_bias_mse_weight: 10.0
- boundary_smoothness_weight: 1.0
- reg_schedule: constant
- reg_start_step: 0
- reg_ramp_steps: 0

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| axis_plus_toric_residual_R0_top55 | centered | interior_only | 48 | 0.5641 | 0.0377 | 0.5766 | 0.4137 | 0.7160 |
| axis_plus_toric_residual_R0_top55 | centered | reflect_padding | 48 | 0.5641 | 0.0336 | 0.5766 | 0.4137 | 0.7160 |
| axis_plus_toric_residual_R0_top55 | normalized | interior_only | 48 | 0.5641 | 0.0377 | 0.5766 | 0.4137 | 0.7160 |
| axis_plus_toric_residual_R0_top55 | normalized | reflect_padding | 48 | 0.5641 | 0.0336 | 0.5766 | 0.4137 | 0.7160 |
| axis_plus_toric_residual_R0_top55 | raw | interior_only | 48 | 0.5517 | 0.0377 | 0.5766 | 0.4137 | 0.7160 |
| axis_plus_toric_residual_R0_top55 | raw | reflect_padding | 48 | 0.5517 | 0.0336 | 0.5766 | 0.4137 | 0.7160 |

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
