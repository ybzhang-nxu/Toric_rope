# V15 NSynth CQT Empirical Table Projection

Records: 256
Token grid: 32x14
Patch dim: 24

| basis | features | R2 | order0 energy | order1 energy | order2 energy | condition |
|---|---:|---:|---:|---:|---:|---:|
| dct_top73 | 73 | 0.9988 | 1.000 | 0.000 | 0.000 | 1.00e+00 |
| dct_top13 | 13 | 0.8968 | 1.000 | 0.000 | 0.000 | 1.00e+00 |
| fft_top6_J2 | 73 | 0.8939 | 73.014 | 31.862 | 49.273 | 2.09e+05 |
| fft_top6_J1 | 37 | 0.8336 | 2.243 | 1.997 | 0.000 | 1.13e+02 |
| fft_top6_J0 | 13 | 0.7630 | 1.000 | 0.000 | 0.000 | 1.00e+00 |
| fft_top6_J2_coord_shuffle | 73 | 0.0410 | 319.968 | 171.446 | 233.727 | 2.09e+05 |
| constant | 1 | 0.0000 | 1.000 | 0.000 | 0.000 | 1.00e+00 |

## Reading

This is a real-data projection diagnostic, not a downstream reconstruction score.
The target is an empirical scalar offset table estimated from CQT patch covariance.
It tests whether the measured NSynth time-frequency geometry has low-dimensional
Toric/PJ structure before asking a small Transformer to exploit it.
Because the PJ columns are non-orthogonal, order-energy entries are contribution
diagnostics rather than normalized variance shares.

Artifacts:

- `projection_results.csv`
- `offset_table.npy`
- `offset_table_heatmap.pdf`
- `projection_r2.pdf`
- `summary.json`
