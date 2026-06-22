# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: toric_PJ_R2
- seed: 443
- steps: 10000
- best_step: 9999
- score: 0.7010729908943176
- final_score: 0.7004276514053345
- train_score: 0.7052909135818481
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 55

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| toric_PJ_R2 | centered | interior_only | 48 | 0.7500 | 0.1290 | 0.6857 | 0.3880 | 0.5618 |
| toric_PJ_R2 | centered | reflect_padding | 48 | 0.7500 | 0.0348 | 0.6857 | 0.3880 | 0.5618 |
| toric_PJ_R2 | normalized | interior_only | 48 | 0.7500 | 0.1290 | 0.6857 | 0.3880 | 0.5618 |
| toric_PJ_R2 | normalized | reflect_padding | 48 | 0.7500 | 0.0348 | 0.6857 | 0.3880 | 0.5618 |
| toric_PJ_R2 | raw | interior_only | 48 | 0.7410 | 0.1290 | 0.6857 | 0.3880 | 0.5618 |
| toric_PJ_R2 | raw | reflect_padding | 48 | 0.7410 | 0.0348 | 0.6857 | 0.3880 | 0.5618 |

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
