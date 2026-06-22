# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: table_informed_toric_PJ_R0_top110
- seed: 477
- steps: 10000
- best_step: 9000
- score: 0.8631461262702942
- final_score: 0.8645698428153992
- train_score: 0.8631942272186279
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 109

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| table_informed_toric_PJ_R0_top110 | centered | interior_only | 48 | 0.4494 | 0.0271 | 0.6741 | 0.2828 | 0.4515 |
| table_informed_toric_PJ_R0_top110 | centered | reflect_padding | 48 | 0.4494 | 0.0236 | 0.6741 | 0.2828 | 0.4515 |
| table_informed_toric_PJ_R0_top110 | normalized | interior_only | 48 | 0.4494 | 0.0271 | 0.6741 | 0.2828 | 0.4515 |
| table_informed_toric_PJ_R0_top110 | normalized | reflect_padding | 48 | 0.4494 | 0.0236 | 0.6741 | 0.2828 | 0.4515 |
| table_informed_toric_PJ_R0_top110 | raw | interior_only | 48 | 0.4165 | 0.0271 | 0.6741 | 0.2828 | 0.4515 |
| table_informed_toric_PJ_R0_top110 | raw | reflect_padding | 48 | 0.4165 | 0.0236 | 0.6741 | 0.2828 | 0.4515 |

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
