# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: pruned_toric_PJ
- seed: 460
- steps: 10000
- best_step: 9999
- score: 0.6765164732933044
- final_score: 0.6774594783782959
- train_score: 0.6806628704071045
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 33

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pruned_toric_PJ | centered | interior_only | 48 | 0.7933 | 0.1217 | 0.6408 | 0.4904 | 0.5894 |
| pruned_toric_PJ | centered | reflect_padding | 48 | 0.7933 | 0.0394 | 0.6408 | 0.4904 | 0.5894 |
| pruned_toric_PJ | normalized | interior_only | 48 | 0.7933 | 0.1217 | 0.6408 | 0.4904 | 0.5894 |
| pruned_toric_PJ | normalized | reflect_padding | 48 | 0.7933 | 0.0394 | 0.6408 | 0.4904 | 0.5894 |
| pruned_toric_PJ | raw | interior_only | 48 | 0.7823 | 0.1217 | 0.6408 | 0.4904 | 0.5894 |
| pruned_toric_PJ | raw | reflect_padding | 48 | 0.7823 | 0.0394 | 0.6408 | 0.4904 | 0.5894 |

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
