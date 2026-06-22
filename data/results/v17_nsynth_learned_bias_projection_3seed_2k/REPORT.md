# V17 NSynth Learned Bias Projection

Projection diagnostics for learned scalar attention-bias tables exported from V14.

## Mean Tables

| source | basis | R2 | target std | condition |
|---|---|---:|---:|---:|
| relative_2d_table_seed426 | constant | 0.0000 | 1.4632e-02 | 1.00e+00 |
| relative_2d_table_seed426 | fft_top6_J0 | 0.8212 | 1.4632e-02 | 1.00e+00 |
| relative_2d_table_seed426 | fft_top6_J1 | 0.8826 | 1.4632e-02 | 9.21e+01 |
| relative_2d_table_seed426 | fft_top6_J2 | 0.9208 | 1.4632e-02 | 1.06e+05 |
| relative_2d_table_seed426 | fft_top6_J2_coord_shuffle | 0.0341 | 1.4632e-02 | 1.06e+05 |
| relative_2d_table_seed426 | dct_top13 | 0.8760 | 1.4632e-02 | 1.00e+00 |
| relative_2d_table_seed426 | dct_top73 | 0.9727 | 1.4632e-02 | 1.00e+00 |
| relative_2d_table_seed526 | constant | 0.0000 | 1.4088e-02 | 1.00e+00 |
| relative_2d_table_seed526 | fft_top6_J0 | 0.8213 | 1.4088e-02 | 1.00e+00 |
| relative_2d_table_seed526 | fft_top6_J1 | 0.8899 | 1.4088e-02 | 1.62e+02 |
| relative_2d_table_seed526 | fft_top6_J2 | 0.9273 | 1.4088e-02 | 4.90e+05 |
| relative_2d_table_seed526 | fft_top6_J2_coord_shuffle | 0.0288 | 1.4088e-02 | 4.89e+05 |
| relative_2d_table_seed526 | dct_top13 | 0.8688 | 1.4088e-02 | 1.00e+00 |
| relative_2d_table_seed526 | dct_top73 | 0.9664 | 1.4088e-02 | 1.00e+00 |
| relative_2d_table_seed626 | constant | 0.0000 | 1.5943e-02 | 1.00e+00 |
| relative_2d_table_seed626 | fft_top6_J0 | 0.8243 | 1.5943e-02 | 1.00e+00 |
| relative_2d_table_seed626 | fft_top6_J1 | 0.9012 | 1.5943e-02 | 1.20e+02 |
| relative_2d_table_seed626 | fft_top6_J2 | 0.9408 | 1.5943e-02 | 1.59e+05 |
| relative_2d_table_seed626 | fft_top6_J2_coord_shuffle | 0.0320 | 1.5943e-02 | 1.59e+05 |
| relative_2d_table_seed626 | dct_top13 | 0.8535 | 1.5943e-02 | 1.00e+00 |
| relative_2d_table_seed626 | dct_top73 | 0.9764 | 1.5943e-02 | 1.00e+00 |

## Mean-Table Margins

| source | comparison | margin | lhs R2 | rhs R2 |
|---|---|---:|---:|---:|
| relative_2d_table_seed426 | J1_minus_J0 | 0.0614 | 0.8826 | 0.8212 |
| relative_2d_table_seed426 | J2_minus_J1 | 0.0382 | 0.9208 | 0.8826 |
| relative_2d_table_seed426 | J2_minus_J0 | 0.0996 | 0.9208 | 0.8212 |
| relative_2d_table_seed426 | J2_minus_shuffle | 0.8867 | 0.9208 | 0.0341 |
| relative_2d_table_seed426 | DCT73_minus_J2 | 0.0518 | 0.9727 | 0.9208 |
| relative_2d_table_seed526 | J1_minus_J0 | 0.0687 | 0.8899 | 0.8213 |
| relative_2d_table_seed526 | J2_minus_J1 | 0.0374 | 0.9273 | 0.8899 |
| relative_2d_table_seed526 | J2_minus_J0 | 0.1060 | 0.9273 | 0.8213 |
| relative_2d_table_seed526 | J2_minus_shuffle | 0.8985 | 0.9273 | 0.0288 |
| relative_2d_table_seed526 | DCT73_minus_J2 | 0.0391 | 0.9664 | 0.9273 |
| relative_2d_table_seed626 | J1_minus_J0 | 0.0769 | 0.9012 | 0.8243 |
| relative_2d_table_seed626 | J2_minus_J1 | 0.0396 | 0.9408 | 0.9012 |
| relative_2d_table_seed626 | J2_minus_J0 | 0.1165 | 0.9408 | 0.8243 |
| relative_2d_table_seed626 | J2_minus_shuffle | 0.9089 | 0.9408 | 0.0320 |
| relative_2d_table_seed626 | DCT73_minus_J2 | 0.0355 | 0.9764 | 0.9408 |

## All-Table Margin Aggregate

This aggregate includes every exported head table plus the mean table.

| comparison | n | mean | std | min | max | positive |
|---|---:|---:|---:|---:|---:|---:|
| DCT73_minus_J2 | 15 | 0.0408 | 0.0175 | 0.0187 | 0.0837 | 15 |
| J1_minus_J0 | 15 | 0.0820 | 0.0151 | 0.0596 | 0.1196 | 15 |
| J2_minus_J0 | 15 | 0.1259 | 0.0142 | 0.0996 | 0.1484 | 15 |
| J2_minus_J1 | 15 | 0.0439 | 0.0087 | 0.0254 | 0.0597 | 15 |
| J2_minus_shuffle | 15 | 0.8935 | 0.0236 | 0.8447 | 0.9282 | 15 |

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
