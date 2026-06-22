# V17 NSynth Learned Bias Projection

Projection diagnostics for learned scalar attention-bias tables exported from V14.

## Mean Tables

| source | basis | R2 | target std | condition |
|---|---|---:|---:|---:|
| relative_2d_table_seed426 | constant | 0.0000 | 1.4632e-02 | 1.00e+00 |
| relative_2d_table_seed426 | fft_top18_J0 | 0.9044 | 1.4632e-02 | 1.00e+00 |
| relative_2d_table_seed426 | fft_top18_J1 | 0.9593 | 1.4632e-02 | 1.68e+07 |
| relative_2d_table_seed426 | fft_top18_J2 | 0.9757 | 1.4632e-02 | 3.29e+07 |
| relative_2d_table_seed426 | fft_top18_J2_coord_shuffle | 0.0759 | 1.4632e-02 | 3.18e+07 |
| relative_2d_table_seed426 | dct_top13 | 0.8760 | 1.4632e-02 | 1.00e+00 |
| relative_2d_table_seed426 | dct_top73 | 0.9727 | 1.4632e-02 | 1.00e+00 |
| relative_2d_table_seed526 | constant | 0.0000 | 1.4088e-02 | 1.00e+00 |
| relative_2d_table_seed526 | fft_top18_J0 | 0.9089 | 1.4088e-02 | 1.00e+00 |
| relative_2d_table_seed526 | fft_top18_J1 | 0.9629 | 1.4088e-02 | 1.37e+07 |
| relative_2d_table_seed526 | fft_top18_J2 | 0.9744 | 1.4088e-02 | 3.15e+07 |
| relative_2d_table_seed526 | fft_top18_J2_coord_shuffle | 0.0755 | 1.4088e-02 | 2.89e+07 |
| relative_2d_table_seed526 | dct_top13 | 0.8688 | 1.4088e-02 | 1.00e+00 |
| relative_2d_table_seed526 | dct_top73 | 0.9664 | 1.4088e-02 | 1.00e+00 |
| relative_2d_table_seed626 | constant | 0.0000 | 1.5943e-02 | 1.00e+00 |
| relative_2d_table_seed626 | fft_top18_J0 | 0.9133 | 1.5943e-02 | 1.00e+00 |
| relative_2d_table_seed626 | fft_top18_J1 | 0.9680 | 1.5943e-02 | 1.31e+07 |
| relative_2d_table_seed626 | fft_top18_J2 | 0.9776 | 1.5943e-02 | 3.63e+07 |
| relative_2d_table_seed626 | fft_top18_J2_coord_shuffle | 0.0786 | 1.5943e-02 | 3.50e+07 |
| relative_2d_table_seed626 | dct_top13 | 0.8535 | 1.5943e-02 | 1.00e+00 |
| relative_2d_table_seed626 | dct_top73 | 0.9764 | 1.5943e-02 | 1.00e+00 |

## Mean-Table Margins

| source | comparison | margin | lhs R2 | rhs R2 |
|---|---|---:|---:|---:|
| relative_2d_table_seed426 | J1_minus_J0 | 0.0550 | 0.9593 | 0.9044 |
| relative_2d_table_seed426 | J2_minus_J1 | 0.0164 | 0.9757 | 0.9593 |
| relative_2d_table_seed426 | J2_minus_J0 | 0.0714 | 0.9757 | 0.9044 |
| relative_2d_table_seed426 | J2_minus_shuffle | 0.8998 | 0.9757 | 0.0759 |
| relative_2d_table_seed426 | DCT73_minus_J2 | -0.0031 | 0.9727 | 0.9757 |
| relative_2d_table_seed526 | J1_minus_J0 | 0.0540 | 0.9629 | 0.9089 |
| relative_2d_table_seed526 | J2_minus_J1 | 0.0115 | 0.9744 | 0.9629 |
| relative_2d_table_seed526 | J2_minus_J0 | 0.0655 | 0.9744 | 0.9089 |
| relative_2d_table_seed526 | J2_minus_shuffle | 0.8989 | 0.9744 | 0.0755 |
| relative_2d_table_seed526 | DCT73_minus_J2 | -0.0080 | 0.9664 | 0.9744 |
| relative_2d_table_seed626 | J1_minus_J0 | 0.0547 | 0.9680 | 0.9133 |
| relative_2d_table_seed626 | J2_minus_J1 | 0.0096 | 0.9776 | 0.9680 |
| relative_2d_table_seed626 | J2_minus_J0 | 0.0643 | 0.9776 | 0.9133 |
| relative_2d_table_seed626 | J2_minus_shuffle | 0.8990 | 0.9776 | 0.0786 |
| relative_2d_table_seed626 | DCT73_minus_J2 | -0.0012 | 0.9764 | 0.9776 |

## All-Table Margin Aggregate

This aggregate includes every exported head table plus the mean table.

| comparison | n | mean | std | min | max | positive |
|---|---:|---:|---:|---:|---:|---:|
| DCT73_minus_J2 | 15 | -0.0080 | 0.0056 | -0.0199 | -0.0007 | 0 |
| J1_minus_J0 | 15 | 0.0633 | 0.0141 | 0.0431 | 0.0857 | 15 |
| J2_minus_J0 | 15 | 0.0810 | 0.0164 | 0.0527 | 0.1156 | 15 |
| J2_minus_J1 | 15 | 0.0177 | 0.0056 | 0.0096 | 0.0300 | 15 |
| J2_minus_shuffle | 15 | 0.8941 | 0.0066 | 0.8829 | 0.9070 | 15 |

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
