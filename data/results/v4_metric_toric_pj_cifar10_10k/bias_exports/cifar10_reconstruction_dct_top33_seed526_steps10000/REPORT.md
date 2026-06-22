# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: dct_top33
- seed: 526
- steps: 10000
- best_step: 9999
- score: 0.8523311018943787
- final_score: 0.8518316149711609
- train_score: 0.8530988693237305
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 34

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| dct_top33 | centered | interior_only | 48 | 0.2604 | 0.0188 | 0.6167 | 0.4490 | 0.5769 |
| dct_top33 | centered | reflect_padding | 48 | 0.2604 | 0.0149 | 0.6167 | 0.4490 | 0.5769 |
| dct_top33 | normalized | interior_only | 48 | 0.2604 | 0.0188 | 0.6167 | 0.4490 | 0.5769 |
| dct_top33 | normalized | reflect_padding | 48 | 0.2604 | 0.0149 | 0.6167 | 0.4490 | 0.5769 |
| dct_top33 | raw | interior_only | 48 | 0.2253 | 0.0188 | 0.6167 | 0.4490 | 0.5769 |
| dct_top33 | raw | reflect_padding | 48 | 0.2253 | 0.0149 | 0.6167 | 0.4490 | 0.5769 |

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
