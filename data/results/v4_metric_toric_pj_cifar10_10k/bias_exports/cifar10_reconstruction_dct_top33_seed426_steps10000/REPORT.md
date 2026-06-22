# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: dct_top33
- seed: 426
- steps: 10000
- best_step: 9999
- score: 0.8438616394996643
- final_score: 0.8436225652694702
- train_score: 0.8449878692626953
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 34

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| dct_top33 | centered | interior_only | 48 | 0.2757 | 0.0197 | 0.6200 | 0.4415 | 0.5779 |
| dct_top33 | centered | reflect_padding | 48 | 0.2757 | 0.0156 | 0.6200 | 0.4415 | 0.5779 |
| dct_top33 | normalized | interior_only | 48 | 0.2757 | 0.0197 | 0.6200 | 0.4415 | 0.5779 |
| dct_top33 | normalized | reflect_padding | 48 | 0.2757 | 0.0156 | 0.6200 | 0.4415 | 0.5779 |
| dct_top33 | raw | interior_only | 48 | 0.2396 | 0.0197 | 0.6200 | 0.4415 | 0.5779 |
| dct_top33 | raw | reflect_padding | 48 | 0.2396 | 0.0156 | 0.6200 | 0.4415 | 0.5779 |

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
