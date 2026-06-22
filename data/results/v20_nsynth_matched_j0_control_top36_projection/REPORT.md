# V17 NSynth Learned Bias Projection

Projection diagnostics for learned scalar attention-bias tables exported from V14.

## Mean Tables

| source | basis | R2 | target std | condition |
|---|---|---:|---:|---:|
| relative_2d_table_seed426 | constant | 0.0000 | 1.4632e-02 | 1.00e+00 |
| relative_2d_table_seed426 | fft_top36_J0 | 0.9415 | 1.4632e-02 | 1.00e+00 |
| relative_2d_table_seed426 | fft_top36_J1 | 0.9818 | 1.4632e-02 | 2.90e+07 |
| relative_2d_table_seed426 | fft_top36_J2 | 0.9909 | 1.4632e-02 | 2.52e+08 |
| relative_2d_table_seed426 | fft_top36_J2_coord_shuffle | 0.1525 | 1.4632e-02 | 3.50e+08 |
| relative_2d_table_seed426 | dct_top13 | 0.8760 | 1.4632e-02 | 1.00e+00 |
| relative_2d_table_seed426 | dct_top73 | 0.9727 | 1.4632e-02 | 1.00e+00 |
| relative_2d_table_seed526 | constant | 0.0000 | 1.4088e-02 | 1.00e+00 |
| relative_2d_table_seed526 | fft_top36_J0 | 0.9415 | 1.4088e-02 | 1.00e+00 |
| relative_2d_table_seed526 | fft_top36_J1 | 0.9788 | 1.4088e-02 | 4.39e+08 |
| relative_2d_table_seed526 | fft_top36_J2 | 0.9887 | 1.4088e-02 | 2.83e+08 |
| relative_2d_table_seed526 | fft_top36_J2_coord_shuffle | 0.1374 | 1.4088e-02 | 7.34e+08 |
| relative_2d_table_seed526 | dct_top13 | 0.8688 | 1.4088e-02 | 1.00e+00 |
| relative_2d_table_seed526 | dct_top73 | 0.9664 | 1.4088e-02 | 1.00e+00 |
| relative_2d_table_seed626 | constant | 0.0000 | 1.5943e-02 | 1.00e+00 |
| relative_2d_table_seed626 | fft_top36_J0 | 0.9398 | 1.5943e-02 | 1.00e+00 |
| relative_2d_table_seed626 | fft_top36_J1 | 0.9857 | 1.5943e-02 | 4.07e+07 |
| relative_2d_table_seed626 | fft_top36_J2 | 0.9925 | 1.5943e-02 | 5.03e+08 |
| relative_2d_table_seed626 | fft_top36_J2_coord_shuffle | 0.1492 | 1.5943e-02 | 2.31e+08 |
| relative_2d_table_seed626 | dct_top13 | 0.8535 | 1.5943e-02 | 1.00e+00 |
| relative_2d_table_seed626 | dct_top73 | 0.9764 | 1.5943e-02 | 1.00e+00 |

## Mean-Table Margins

| source | comparison | margin | lhs R2 | rhs R2 |
|---|---|---:|---:|---:|
| relative_2d_table_seed426 | J1_minus_J0 | 0.0403 | 0.9818 | 0.9415 |
| relative_2d_table_seed426 | J2_minus_J1 | 0.0091 | 0.9909 | 0.9818 |
| relative_2d_table_seed426 | J2_minus_J0 | 0.0494 | 0.9909 | 0.9415 |
| relative_2d_table_seed426 | J2_minus_shuffle | 0.8384 | 0.9909 | 0.1525 |
| relative_2d_table_seed426 | DCT73_minus_J2 | -0.0182 | 0.9727 | 0.9909 |
| relative_2d_table_seed526 | J1_minus_J0 | 0.0373 | 0.9788 | 0.9415 |
| relative_2d_table_seed526 | J2_minus_J1 | 0.0098 | 0.9887 | 0.9788 |
| relative_2d_table_seed526 | J2_minus_J0 | 0.0471 | 0.9887 | 0.9415 |
| relative_2d_table_seed526 | J2_minus_shuffle | 0.8512 | 0.9887 | 0.1374 |
| relative_2d_table_seed526 | DCT73_minus_J2 | -0.0222 | 0.9664 | 0.9887 |
| relative_2d_table_seed626 | J1_minus_J0 | 0.0458 | 0.9857 | 0.9398 |
| relative_2d_table_seed626 | J2_minus_J1 | 0.0069 | 0.9925 | 0.9857 |
| relative_2d_table_seed626 | J2_minus_J0 | 0.0527 | 0.9925 | 0.9398 |
| relative_2d_table_seed626 | J2_minus_shuffle | 0.8433 | 0.9925 | 0.1492 |
| relative_2d_table_seed626 | DCT73_minus_J2 | -0.0161 | 0.9764 | 0.9925 |

## All-Table Margin Aggregate

This aggregate includes every exported head table plus the mean table.

| comparison | n | mean | std | min | max | positive |
|---|---:|---:|---:|---:|---:|---:|
| DCT73_minus_J2 | 15 | -0.0119 | 0.0416 | -0.0392 | 0.1411 | 1 |
| J1_minus_J0 | 15 | 0.0505 | 0.0106 | 0.0308 | 0.0651 | 15 |
| J2_minus_J0 | 15 | 0.0479 | 0.0397 | -0.0950 | 0.0769 | 14 |
| J2_minus_J1 | 15 | -0.0026 | 0.0422 | -0.1601 | 0.0144 | 14 |
| J2_minus_shuffle | 15 | 0.8272 | 0.0463 | 0.6554 | 0.8512 | 15 |

## Reading

This diagnostic asks whether a learned relative-table scalar-bias teacher
contains the same high-order structure seen in empirical CQT offset-table
projection.  It is sensitive to whether the downstream teacher actually
learned a nontrivial positional table.

Artifacts:

- `learned_bias_projection_rows.csv`
- `learned_bias_projection_margins.csv`
- `learned_bias_projection_margins_aggregate.csv`
- `learned_bias_projection_mean_r2.pdf`
- `summary.json`
