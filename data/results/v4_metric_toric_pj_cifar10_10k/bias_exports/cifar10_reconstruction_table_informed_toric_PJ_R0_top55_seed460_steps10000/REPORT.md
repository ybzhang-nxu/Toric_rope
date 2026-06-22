# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: table_informed_toric_PJ_R0_top55
- seed: 460
- steps: 10000
- best_step: 9999
- score: 0.8133466243743896
- final_score: 0.8138481974601746
- train_score: 0.8147424459457397
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 55

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| table_informed_toric_PJ_R0_top55 | centered | interior_only | 48 | 0.1782 | 0.0092 | 0.5254 | 0.5001 | 0.6880 |
| table_informed_toric_PJ_R0_top55 | centered | reflect_padding | 48 | 0.1782 | 0.0077 | 0.5254 | 0.5001 | 0.6880 |
| table_informed_toric_PJ_R0_top55 | normalized | interior_only | 48 | 0.1782 | 0.0092 | 0.5254 | 0.5001 | 0.6880 |
| table_informed_toric_PJ_R0_top55 | normalized | reflect_padding | 48 | 0.1782 | 0.0077 | 0.5254 | 0.5001 | 0.6880 |
| table_informed_toric_PJ_R0_top55 | raw | interior_only | 48 | 0.1559 | 0.0092 | 0.5254 | 0.5001 | 0.6880 |
| table_informed_toric_PJ_R0_top55 | raw | reflect_padding | 48 | 0.1559 | 0.0077 | 0.5254 | 0.5001 | 0.6880 |

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
