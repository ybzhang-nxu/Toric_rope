# V12 E6 Resolution Transfer Report

This is a fixed-weight evaluation. Transformer weights and positional coefficients are loaded from existing checkpoints; only the input resize/grid and positional-bias extension rule change.

## Summary

- Dataset: tiny-imagenet
- Train resolution / patch size / grid: 32 / 4 / 8
- Eval resolutions: [32, 40, 48, 64]
- State: best
- Test limit: None

## Aggregate

| method | extension | resolution | grid | n | score mean | score std | far bias RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dct_top110 | functional | 32 | 8 | 3 | 0.8611 | 0.0048 | 0.0207 |
| dct_top110 | functional | 40 | 10 | 3 | 0.8170 | 0.0018 | 0.0217 |
| dct_top110 | functional | 48 | 12 | 3 | 0.7607 | 0.0090 | 0.0229 |
| dct_top110 | functional | 64 | 16 | 3 | 0.6222 | 0.0210 | 0.0263 |
| dct_top110 | no_pos | 32 | 8 | 3 | 0.2838 | 0.0082 | 0.0000 |
| dct_top110 | no_pos | 40 | 10 | 3 | 0.2805 | 0.0104 | 0.0000 |
| dct_top110 | no_pos | 48 | 12 | 3 | 0.2785 | 0.0084 | 0.0000 |
| dct_top110 | no_pos | 64 | 16 | 3 | 0.2503 | 0.0072 | 0.0000 |
| dct_top110 | radial_truncate_r4 | 32 | 8 | 3 | 0.8633 | 0.0038 | 0.0000 |
| dct_top110 | radial_truncate_r4 | 40 | 10 | 3 | 0.8194 | 0.0014 | 0.0000 |
| dct_top110 | radial_truncate_r4 | 48 | 12 | 3 | 0.7614 | 0.0110 | 0.0000 |
| dct_top110 | radial_truncate_r4 | 64 | 16 | 3 | 0.6292 | 0.0169 | 0.0000 |
| dct_top110 | table_clamp | 32 | 8 | 3 | 0.8612 | 0.0048 | 0.0207 |
| dct_top110 | table_clamp | 40 | 10 | 3 | 0.8555 | 0.0125 | 0.0222 |
| dct_top110 | table_clamp | 48 | 12 | 3 | 0.8339 | 0.0258 | 0.0228 |
| dct_top110 | table_clamp | 64 | 16 | 3 | 0.6765 | 0.0608 | 0.0235 |
| farband_repair | functional | 32 | 8 | 3 | 0.8674 | 0.0095 | 0.0399 |
| farband_repair | functional | 40 | 10 | 3 | 0.8687 | 0.0111 | 0.0483 |
| farband_repair | functional | 48 | 12 | 3 | 0.8626 | 0.0127 | 0.0495 |
| farband_repair | functional | 64 | 16 | 3 | 0.7129 | 0.0163 | 0.0959 |
| farband_repair | no_pos | 32 | 8 | 3 | 0.2701 | 0.0101 | 0.0000 |
| farband_repair | no_pos | 40 | 10 | 3 | 0.2626 | 0.0106 | 0.0000 |
| farband_repair | no_pos | 48 | 12 | 3 | 0.2565 | 0.0102 | 0.0000 |
| farband_repair | no_pos | 64 | 16 | 3 | 0.2299 | 0.0114 | 0.0000 |
| farband_repair | radial_truncate_r4 | 32 | 8 | 3 | 0.8729 | 0.0047 | 0.0000 |
| farband_repair | radial_truncate_r4 | 40 | 10 | 3 | 0.8735 | 0.0066 | 0.0000 |
| farband_repair | radial_truncate_r4 | 48 | 12 | 3 | 0.8672 | 0.0078 | 0.0000 |
| farband_repair | radial_truncate_r4 | 64 | 16 | 3 | 0.7505 | 0.0147 | 0.0000 |
| farband_repair | table_clamp | 32 | 8 | 3 | 0.8674 | 0.0095 | 0.0399 |
| farband_repair | table_clamp | 40 | 10 | 3 | 0.8693 | 0.0116 | 0.0482 |
| farband_repair | table_clamp | 48 | 12 | 3 | 0.8633 | 0.0152 | 0.0531 |
| farband_repair | table_clamp | 64 | 16 | 3 | 0.7373 | 0.0327 | 0.0594 |
| relative_table | functional | 32 | 8 | 3 | 0.8380 | 0.0029 | 0.0000 |
| relative_table | functional | 40 | 10 | 3 | 0.8085 | 0.0036 | 0.0123 |
| relative_table | functional | 48 | 12 | 3 | 0.7715 | 0.0028 | 0.0216 |
| relative_table | functional | 64 | 16 | 3 | 0.6518 | 0.0019 | 0.0298 |
| relative_table | no_pos | 32 | 8 | 3 | 0.2409 | 0.0020 | 0.0000 |
| relative_table | no_pos | 40 | 10 | 3 | 0.2404 | 0.0062 | 0.0000 |
| relative_table | no_pos | 48 | 12 | 3 | 0.2414 | 0.0047 | 0.0000 |
| relative_table | no_pos | 64 | 16 | 3 | 0.2155 | 0.0049 | 0.0000 |
| relative_table | radial_truncate_r4 | 32 | 8 | 3 | 0.8377 | 0.0029 | 0.0000 |
| relative_table | radial_truncate_r4 | 40 | 10 | 3 | 0.8074 | 0.0036 | 0.0000 |
| relative_table | radial_truncate_r4 | 48 | 12 | 3 | 0.7720 | 0.0030 | 0.0000 |
| relative_table | radial_truncate_r4 | 64 | 16 | 3 | 0.6589 | 0.0029 | 0.0000 |
| relative_table | table_clamp | 32 | 8 | 3 | 0.8380 | 0.0029 | 0.0000 |
| relative_table | table_clamp | 40 | 10 | 3 | 0.8310 | 0.0056 | 0.0000 |
| relative_table | table_clamp | 48 | 12 | 3 | 0.7932 | 0.0103 | 0.0000 |
| relative_table | table_clamp | 64 | 16 | 3 | 0.5831 | 0.0129 | 0.0000 |
| toric_pj | functional | 32 | 8 | 3 | 0.8250 | 0.0145 | 0.4506 |
| toric_pj | functional | 40 | 10 | 3 | 0.8075 | 0.0204 | 0.4061 |
| toric_pj | functional | 48 | 12 | 3 | 0.7953 | 0.0216 | 0.3991 |
| toric_pj | functional | 64 | 16 | 3 | 0.5964 | 0.0156 | 0.3556 |
| toric_pj | no_pos | 32 | 8 | 3 | 0.2776 | 0.0046 | 0.0000 |
| toric_pj | no_pos | 40 | 10 | 3 | 0.2729 | 0.0041 | 0.0000 |
| toric_pj | no_pos | 48 | 12 | 3 | 0.2691 | 0.0050 | 0.0000 |
| toric_pj | no_pos | 64 | 16 | 3 | 0.2421 | 0.0040 | 0.0000 |
| toric_pj | radial_truncate_r4 | 32 | 8 | 3 | 0.8628 | 0.0021 | 0.0000 |
| toric_pj | radial_truncate_r4 | 40 | 10 | 3 | 0.8646 | 0.0010 | 0.0000 |
| toric_pj | radial_truncate_r4 | 48 | 12 | 3 | 0.8573 | 0.0022 | 0.0000 |
| toric_pj | radial_truncate_r4 | 64 | 16 | 3 | 0.7389 | 0.0039 | 0.0000 |
| toric_pj | table_clamp | 32 | 8 | 3 | 0.8250 | 0.0145 | 0.4506 |
| toric_pj | table_clamp | 40 | 10 | 3 | 0.8086 | 0.0198 | 0.4176 |
| toric_pj | table_clamp | 48 | 12 | 3 | 0.7922 | 0.0218 | 0.3992 |
| toric_pj | table_clamp | 64 | 16 | 3 | 0.6638 | 0.0229 | 0.3746 |

## Claim Boundary

- Resize distribution shift is shared by all methods and controls; do not attribute all score changes to positional encoding.
- `functional` is the native compact-basis extension, while `table_clamp` extends each checkpoint's learned train-window bias table by nearest boundary clamping.
- For `relative_2d_table`, `functional` is scaled bilinear interpolation from the train-window table.
