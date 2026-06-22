# V16 NSynth CQT Projection Stability

Records loaded: 256
Subset sizes: [128, 192, 256]
Subset seeds: [101, 202, 303, 404, 505]

## Projection R2 Aggregate

| subset | basis | n | R2 mean | R2 std | R2 min | R2 max |
|---:|---|---:|---:|---:|---:|---:|
| 128 | constant | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 128 | dct_top13 | 5 | 0.8928 | 0.0148 | 0.8711 | 0.9162 |
| 128 | dct_top73 | 5 | 0.9987 | 0.0001 | 0.9985 | 0.9989 |
| 128 | fft_top6_J0 | 5 | 0.7581 | 0.0202 | 0.7373 | 0.7968 |
| 128 | fft_top6_J1 | 5 | 0.8316 | 0.0180 | 0.8163 | 0.8666 |
| 128 | fft_top6_J2 | 5 | 0.8928 | 0.0144 | 0.8792 | 0.9206 |
| 128 | fft_top6_J2_coord_shuffle | 5 | -0.0405 | 0.0996 | -0.2346 | 0.0246 |
| 192 | constant | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 192 | dct_top13 | 5 | 0.8881 | 0.0118 | 0.8683 | 0.8984 |
| 192 | dct_top73 | 5 | 0.9987 | 0.0001 | 0.9985 | 0.9989 |
| 192 | fft_top6_J0 | 5 | 0.7532 | 0.0130 | 0.7378 | 0.7693 |
| 192 | fft_top6_J1 | 5 | 0.8267 | 0.0105 | 0.8127 | 0.8415 |
| 192 | fft_top6_J2 | 5 | 0.8892 | 0.0080 | 0.8799 | 0.9002 |
| 192 | fft_top6_J2_coord_shuffle | 5 | -0.0266 | 0.1121 | -0.2507 | 0.0333 |
| 256 | constant | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 256 | dct_top13 | 1 | 0.8968 | 0.0000 | 0.8968 | 0.8968 |
| 256 | dct_top73 | 1 | 0.9988 | 0.0000 | 0.9988 | 0.9988 |
| 256 | fft_top6_J0 | 1 | 0.7630 | 0.0000 | 0.7630 | 0.7630 |
| 256 | fft_top6_J1 | 1 | 0.8336 | 0.0000 | 0.8336 | 0.8336 |
| 256 | fft_top6_J2 | 1 | 0.8939 | 0.0000 | 0.8939 | 0.8939 |
| 256 | fft_top6_J2_coord_shuffle | 1 | 0.0238 | 0.0000 | 0.0238 | 0.0238 |

## Key Margins

| subset | comparison | n | margin mean | margin std | margin min | margin max |
|---:|---|---:|---:|---:|---:|---:|
| 128 | DCT73_minus_J2 | 5 | 0.1059 | 0.0143 | 0.0783 | 0.1193 |
| 128 | J1_minus_J0 | 5 | 0.0735 | 0.0036 | 0.0697 | 0.0789 |
| 128 | J2_minus_J1 | 5 | 0.0612 | 0.0041 | 0.0541 | 0.0667 |
| 128 | J2_minus_shuffle | 5 | 0.9333 | 0.1133 | 0.8559 | 1.1552 |
| 192 | DCT73_minus_J2 | 5 | 0.1095 | 0.0078 | 0.0987 | 0.1186 |
| 192 | J1_minus_J0 | 5 | 0.0735 | 0.0033 | 0.0695 | 0.0791 |
| 192 | J2_minus_J1 | 5 | 0.0624 | 0.0029 | 0.0587 | 0.0675 |
| 192 | J2_minus_shuffle | 5 | 0.9158 | 0.1177 | 0.8483 | 1.1509 |
| 256 | DCT73_minus_J2 | 1 | 0.1049 | 0.0000 | 0.1049 | 0.1049 |
| 256 | J1_minus_J0 | 1 | 0.0706 | 0.0000 | 0.0706 | 0.0706 |
| 256 | J2_minus_J1 | 1 | 0.0603 | 0.0000 | 0.0603 | 0.0603 |
| 256 | J2_minus_shuffle | 1 | 0.8701 | 0.0000 | 0.8701 | 0.8701 |

## Reading

This is a stability check for the V15 empirical offset-table projection result.
Each subset recomputes the empirical CQT offset table and table-informed FFT
frequencies.  It is still a projection diagnostic, not a downstream task score.

Artifacts:

- `projection_stability_rows.csv`
- `projection_stability_aggregate.csv`
- `projection_stability_margins.csv`
- `projection_stability_margin_aggregate.csv`
- `projection_stability_margins.pdf`
- `summary.json`
