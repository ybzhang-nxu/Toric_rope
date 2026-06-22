# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: classification
- basis: relative_2d_table
- seed: 594
- steps: 5000
- best_step: 2000
- score: 0.5789999961853027
- final_score: 0.5769999623298645
- train_score: 1.0
- grid_side: 8
- n_layers: 4
- n_heads: 8
- num_features: 225

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| relative_2d_table | centered | interior_only | 32 | 0.3995 | 0.0188 | 0.6971 | 0.4649 | 0.5778 |
| relative_2d_table | centered | reflect_padding | 32 | 0.3995 | 0.0103 | 0.6971 | 0.4649 | 0.5778 |
| relative_2d_table | normalized | interior_only | 32 | 0.3995 | 0.0188 | 0.6971 | 0.4649 | 0.5778 |
| relative_2d_table | normalized | reflect_padding | 32 | 0.3995 | 0.0103 | 0.6971 | 0.4649 | 0.5778 |
| relative_2d_table | raw | interior_only | 32 | 0.3615 | 0.0188 | 0.6971 | 0.4649 | 0.5778 |
| relative_2d_table | raw | reflect_padding | 32 | 0.3615 | 0.0103 | 0.6971 | 0.4649 | 0.5778 |

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
