# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: table_informed_toric_PJ_R0_top110
- seed: 577
- steps: 10000
- best_step: 9999
- score: 0.8651331663131714
- final_score: 0.8665660619735718
- train_score: 0.8660142421722412
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 109

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| table_informed_toric_PJ_R0_top110 | centered | interior_only | 48 | 0.4376 | 0.0243 | 0.6775 | 0.2774 | 0.4286 |
| table_informed_toric_PJ_R0_top110 | centered | reflect_padding | 48 | 0.4376 | 0.0215 | 0.6775 | 0.2774 | 0.4286 |
| table_informed_toric_PJ_R0_top110 | normalized | interior_only | 48 | 0.4376 | 0.0243 | 0.6775 | 0.2774 | 0.4286 |
| table_informed_toric_PJ_R0_top110 | normalized | reflect_padding | 48 | 0.4376 | 0.0215 | 0.6775 | 0.2774 | 0.4286 |
| table_informed_toric_PJ_R0_top110 | raw | interior_only | 48 | 0.4072 | 0.0243 | 0.6775 | 0.2774 | 0.4286 |
| table_informed_toric_PJ_R0_top110 | raw | reflect_padding | 48 | 0.4072 | 0.0215 | 0.6775 | 0.2774 | 0.4286 |

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
