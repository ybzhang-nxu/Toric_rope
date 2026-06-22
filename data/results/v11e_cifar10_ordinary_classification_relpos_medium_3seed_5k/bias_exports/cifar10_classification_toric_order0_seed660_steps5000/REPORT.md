# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: classification
- basis: toric_order0
- seed: 660
- steps: 5000
- best_step: 4000
- score: 0.5745999813079834
- final_score: 0.5735999941825867
- train_score: 1.0
- grid_side: 8
- n_layers: 4
- n_heads: 8
- num_features: 7

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| toric_order0 | centered | interior_only | 32 | 0.9550 | 0.1539 | 0.4928 | 0.6901 | 0.6973 |
| toric_order0 | centered | reflect_padding | 32 | 0.9550 | 0.0819 | 0.4928 | 0.6901 | 0.6973 |
| toric_order0 | normalized | interior_only | 32 | 0.9550 | 0.1539 | 0.4928 | 0.6901 | 0.6973 |
| toric_order0 | normalized | reflect_padding | 32 | 0.9550 | 0.0819 | 0.4928 | 0.6901 | 0.6973 |
| toric_order0 | raw | interior_only | 32 | 0.9394 | 0.1539 | 0.4928 | 0.6901 | 0.6973 |
| toric_order0 | raw | reflect_padding | 32 | 0.9394 | 0.0819 | 0.4928 | 0.6901 | 0.6973 |

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
