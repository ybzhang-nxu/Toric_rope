# V4 Relative Table Geometry Report

Metadata:

- dataset: cifar10
- task: reconstruction
- basis: dct_top55
- seed: 443
- steps: 10000
- best_step: 9999
- score: 0.8275274038314819
- final_score: 0.8278971910476685
- train_score: 0.8285194039344788
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 56

## Aggregate Geometry

| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| dct_top55 | centered | interior_only | 48 | 0.3847 | 0.0323 | 0.6960 | 0.3773 | 0.5061 |
| dct_top55 | centered | reflect_padding | 48 | 0.3847 | 0.0270 | 0.6960 | 0.3773 | 0.5061 |
| dct_top55 | normalized | interior_only | 48 | 0.3847 | 0.0323 | 0.6960 | 0.3773 | 0.5061 |
| dct_top55 | normalized | reflect_padding | 48 | 0.3847 | 0.0270 | 0.6960 | 0.3773 | 0.5061 |
| dct_top55 | raw | interior_only | 48 | 0.3210 | 0.0323 | 0.6960 | 0.3773 | 0.5061 |
| dct_top55 | raw | reflect_padding | 48 | 0.3210 | 0.0270 | 0.6960 | 0.3773 | 0.5061 |

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
