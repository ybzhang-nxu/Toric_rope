# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: dct_top33
- seed: 526
- steps: 10000
- best_step: 9999
- score: 0.63894122838974
- final_score: 0.6379877328872681
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 34
- holdout_radius: 4
- heldout_bias_mse_weight: 1.0
- boundary_smoothness_weight: 10.0
- reg_schedule: constant
- reg_start_step: 0
- reg_ramp_steps: 0

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| dct_top33 | centered | interior_only | 48 | 0.7321 | 0.0103 | 0.6881 | 0.3173 | 0.6695 |
| dct_top33 | centered | reflect_padding | 48 | 0.7321 | 0.0079 | 0.6881 | 0.3173 | 0.6695 |
| dct_top33 | normalized | interior_only | 48 | 0.7321 | 0.0103 | 0.6881 | 0.3173 | 0.6695 |
| dct_top33 | normalized | reflect_padding | 48 | 0.7321 | 0.0079 | 0.6881 | 0.3173 | 0.6695 |
| dct_top33 | raw | interior_only | 48 | 0.7285 | 0.0103 | 0.6881 | 0.3173 | 0.6695 |
| dct_top33 | raw | reflect_padding | 48 | 0.7285 | 0.0079 | 0.6881 | 0.3173 | 0.6695 |

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
