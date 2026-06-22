# V12 E0 Controlled Sector Recovery

Rows: 134400
Cases: 400

## Noise-free full-window R2 by target family

| target family | model | n | full R2 mean | ext R2 mean | heldout RMS mean | condition mean |
|---|---|---:|---:|---:|---:|---:|
| J0 | axis_j0 | 2 | 0.1080 | -0.4282 | 0.7169 | 2.44e+01 |
| J0 | coord_shuffle_j2 | 2 | -0.1620 | -0.2203 | 0.8038 | 2.67e+02 |
| J0 | dct_matched | 2 | -0.0581 | -0.8402 | 0.8935 | 1.00e+00 |
| J0 | directional_j1 | 2 | 1.0000 | 1.0000 | 0.0000 | 2.58e+00 |
| J0 | directional_j2 | 2 | 1.0000 | 1.0000 | 0.0000 | 5.47e+00 |
| J0 | full_j1 | 2 | 1.0000 | 1.0000 | 0.0000 | 7.19e+00 |
| J0 | full_j2 | 2 | 1.0000 | 1.0000 | 0.0000 | 2.20e+02 |
| J0 | toric_j0 | 2 | 1.0000 | 1.0000 | 0.0000 | 1.40e+00 |
| J1 | axis_j0 | 3 | -0.0558 | -0.0437 | 0.9198 | 2.44e+01 |
| J1 | coord_shuffle_j2 | 3 | -0.0414 | -0.0127 | 0.9116 | 2.67e+02 |
| J1 | dct_matched | 3 | 0.0022 | -0.0406 | 0.9394 | 1.00e+00 |
| J1 | directional_j1 | 3 | 0.5156 | 0.5119 | 0.4356 | 2.58e+00 |
| J1 | directional_j2 | 3 | 0.0522 | -5.5759 | 0.5168 | 5.47e+00 |
| J1 | full_j1 | 3 | 1.0000 | 1.0000 | 0.0000 | 7.19e+00 |
| J1 | full_j2 | 3 | 1.0000 | 1.0000 | 0.0000 | 2.20e+02 |
| J1 | toric_j0 | 3 | -0.0606 | -0.0193 | 0.9095 | 1.40e+00 |
| J2 | axis_j0 | 3 | 0.0087 | -0.0252 | 1.2639 | 2.44e+01 |
| J2 | coord_shuffle_j2 | 3 | -0.0501 | -0.0053 | 1.2988 | 2.67e+02 |
| J2 | dct_matched | 3 | 0.0042 | -0.0195 | 1.2834 | 1.00e+00 |
| J2 | directional_j1 | 3 | 0.1542 | 0.0361 | 1.1422 | 2.58e+00 |
| J2 | directional_j2 | 3 | 0.4428 | 0.3205 | 0.6117 | 5.47e+00 |
| J2 | full_j1 | 3 | 0.1193 | 0.0040 | 1.1487 | 7.19e+00 |
| J2 | full_j2 | 3 | 1.0000 | 1.0000 | 0.0000 | 2.20e+02 |
| J2 | toric_j0 | 3 | 0.1773 | 0.0428 | 1.1340 | 1.40e+00 |
| affine | axis_j0 | 3 | 0.1515 | 0.0497 | 1.1584 | 2.44e+01 |
| affine | coord_shuffle_j2 | 3 | -0.0473 | -0.0100 | 1.3532 | 2.67e+02 |
| affine | dct_matched | 3 | 0.6695 | -0.1925 | 0.7908 | 1.00e+00 |
| affine | directional_j1 | 3 | -0.0781 | -0.1509 | 1.3730 | 2.58e+00 |
| affine | directional_j2 | 3 | -0.8562 | -5.3786 | 1.6498 | 5.47e+00 |
| affine | full_j1 | 3 | -0.0651 | -0.2605 | 1.3577 | 7.19e+00 |
| affine | full_j2 | 3 | -0.7613 | -5.1171 | 1.6805 | 2.20e+02 |
| affine | toric_j0 | 3 | 0.0097 | -0.0135 | 1.3177 | 1.40e+00 |
| mixture | axis_j0 | 3 | 0.1911 | -0.0914 | 0.4886 | 2.44e+01 |
| mixture | coord_shuffle_j2 | 3 | -0.1617 | -0.0609 | 0.5827 | 2.67e+02 |
| mixture | dct_matched | 3 | 0.0776 | -0.1952 | 0.5763 | 1.00e+00 |
| mixture | directional_j1 | 3 | 0.5870 | 0.2669 | 0.2802 | 2.58e+00 |
| mixture | directional_j2 | 3 | 0.2780 | -1.7706 | 0.3190 | 5.47e+00 |
| mixture | full_j1 | 3 | 0.5745 | 0.2046 | 0.2800 | 7.19e+00 |
| mixture | full_j2 | 3 | 0.4707 | -1.0406 | 0.2663 | 2.20e+02 |
| mixture | toric_j0 | 3 | 0.5492 | 0.1409 | 0.3705 | 1.40e+00 |

Artifacts:

- `sector_recovery_results.csv`
- `sector_recovery_aggregate.csv`
- `ridge_path.csv`
- `sector_recovery_matrix.pdf`
- `tail_scatter.pdf`
