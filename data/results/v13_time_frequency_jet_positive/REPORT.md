# V13 Time-Frequency High-Order Jet Positive Control

Rows: 3888
Cases: 24

This controlled experiment uses a rectangular time-frequency displacement lattice.
The target functions are carrier, envelope, mixed-envelope, and packet fields of the form
`p(t,f) cos(omega_t t + omega_f f)` or `p(t,f) sin(...)`, where `p` has degree 0, 1, or 2.
The matched DCT baseline uses its top atoms from the clean visible field, so it is a favorable
finite-window compression control rather than a weakened straw baseline.

## Noise-Free Family Summary

| target family | model | n | features | full R2 | ext R2 | heldout RMS | condition |
|---|---|---:|---:|---:|---:|---:|---:|
| J0 | axis_j0 | 24 | 5 | -0.0540 | -0.1047 | 0.7680 | 1.93e+00 |
| J0 | coord_shuffle_full_j2 | 24 | 13 | -0.0057 | -0.0087 | 0.7148 | 3.83e+00 |
| J0 | dct_matched13 | 24 | 13 | -0.0356 | -0.5010 | 0.9662 | 1.00e+00 |
| J0 | directional_j2_one | 24 | 7 | 1.0000 | 1.0000 | 0.0000 | 2.44e+00 |
| J0 | full_j1 | 24 | 7 | 1.0000 | 1.0000 | 0.0000 | 1.10e+00 |
| J0 | full_j2 | 24 | 13 | 1.0000 | 1.0000 | 0.0000 | 3.64e+00 |
| J0 | toric_j0_cluster_matched13 | 24 | 13 | 1.0000 | 1.0000 | 0.0000 | 3.12e+01 |
| J0 | toric_j0_generic_matched13 | 24 | 13 | 0.0684 | -0.0252 | 0.7220 | 1.20e+00 |
| J0 | toric_j0_single | 24 | 3 | 1.0000 | 1.0000 | 0.0000 | 1.04e+00 |
| J1 | axis_j0 | 72 | 5 | -0.0002 | -0.0232 | 0.7746 | 1.93e+00 |
| J1 | coord_shuffle_full_j2 | 72 | 13 | -0.0040 | -0.0042 | 0.7693 | 3.83e+00 |
| J1 | dct_matched13 | 72 | 13 | -0.0755 | -0.1173 | 0.8814 | 1.00e+00 |
| J1 | directional_j2_one | 72 | 7 | 0.6653 | 0.6647 | 0.3154 | 2.44e+00 |
| J1 | full_j1 | 72 | 7 | 1.0000 | 1.0000 | 0.0000 | 1.10e+00 |
| J1 | full_j2 | 72 | 13 | 1.0000 | 1.0000 | 0.0000 | 3.64e+00 |
| J1 | toric_j0_cluster_matched13 | 72 | 13 | 0.9857 | 0.9405 | 0.1017 | 3.12e+01 |
| J1 | toric_j0_generic_matched13 | 72 | 13 | -0.0105 | -0.0010 | 0.7807 | 1.20e+00 |
| J1 | toric_j0_single | 72 | 3 | -0.0010 | -0.0005 | 0.7664 | 1.04e+00 |
| J2 | axis_j0 | 96 | 5 | -0.0199 | -0.0129 | 0.9507 | 1.93e+00 |
| J2 | coord_shuffle_full_j2 | 96 | 13 | -0.0047 | -0.0018 | 0.9393 | 3.83e+00 |
| J2 | dct_matched13 | 96 | 13 | -0.0435 | -0.0263 | 0.9955 | 1.00e+00 |
| J2 | directional_j2_one | 96 | 7 | 0.6218 | 0.5715 | 0.4390 | 2.44e+00 |
| J2 | full_j1 | 96 | 7 | 0.2800 | 0.1821 | 0.7871 | 1.10e+00 |
| J2 | full_j2 | 96 | 13 | 1.0000 | 1.0000 | 0.0000 | 3.64e+00 |
| J2 | toric_j0_cluster_matched13 | 96 | 13 | 0.9091 | 0.8277 | 0.1925 | 3.12e+01 |
| J2 | toric_j0_generic_matched13 | 96 | 13 | 0.0162 | -0.0043 | 0.9345 | 1.20e+00 |
| J2 | toric_j0_single | 96 | 3 | 0.2809 | 0.1827 | 0.7866 | 1.04e+00 |
| mixture | axis_j0 | 24 | 5 | -0.0590 | -0.0956 | 0.4704 | 1.93e+00 |
| mixture | coord_shuffle_full_j2 | 24 | 13 | -0.0060 | -0.0048 | 0.4443 | 3.83e+00 |
| mixture | dct_matched13 | 24 | 13 | 0.0002 | -0.2250 | 0.5407 | 1.00e+00 |
| mixture | directional_j2_one | 24 | 7 | 0.8332 | 0.7224 | 0.1971 | 2.44e+00 |
| mixture | full_j1 | 24 | 7 | 0.9209 | 0.8263 | 0.1471 | 1.10e+00 |
| mixture | full_j2 | 24 | 13 | 1.0000 | 1.0000 | 0.0000 | 3.64e+00 |
| mixture | toric_j0_cluster_matched13 | 24 | 13 | 0.9686 | 0.8644 | 0.0957 | 3.12e+01 |
| mixture | toric_j0_generic_matched13 | 24 | 13 | 0.0517 | -0.0104 | 0.4448 | 1.20e+00 |
| mixture | toric_j0_single | 24 | 3 | 0.6161 | 0.4455 | 0.3067 | 1.04e+00 |

## Key Positive Margins

| target | comparison | full R2 margin | ext R2 margin | heldout RMS improvement |
|---|---|---:|---:|---:|
| j1_time_envelope | full_j1_minus_j0_cluster | 0.0114 | 0.0482 | 0.0918 |
| j1_time_envelope | full_j2_minus_j0_cluster | 0.0114 | 0.0482 | 0.0918 |
| j1_time_envelope | full_j2_minus_dct | 1.0736 | 1.1314 | 0.8822 |
| j2_directional_packet | full_j1_minus_j0_cluster | -0.5914 | -0.6595 | -0.6149 |
| j2_directional_packet | full_j2_minus_j0_cluster | 0.0574 | 0.1141 | 0.2392 |
| j2_directional_packet | full_j2_minus_dct | 1.0012 | 1.0052 | 1.0928 |
| j2_mixed_envelope | full_j1_minus_j0_cluster | -0.7117 | -0.4619 | -0.3242 |
| j2_mixed_envelope | full_j2_minus_j0_cluster | 0.2995 | 0.5407 | 0.4112 |
| j2_mixed_envelope | full_j2_minus_dct | 1.1033 | 1.0485 | 0.7967 |
| mix_j0_j1_j2 | full_j1_minus_j0_cluster | -0.0477 | -0.0381 | -0.0514 |
| mix_j0_j1_j2 | full_j2_minus_j0_cluster | 0.0314 | 0.1356 | 0.0957 |
| mix_j0_j1_j2 | full_j2_minus_dct | 0.9998 | 1.2250 | 0.5407 |

## Reading

The positive result is deliberately controlled.  On true degree-1 and degree-2
time-frequency carrier fields, the matching full PJ order recovers the target
on the visible, full, and extended lattices.  Matched-atom J0 controls and DCT
baselines can fit some visible-window structure but do not represent the
polynomially modulated carrier as cleanly under full/extended evaluation.
The 1% noise setting preserves the same ordering for the main J1, J2, and
mixture targets.

Safe wording:

```text
A controlled time-frequency lattice confirms that two-dimensional higher-order
PJ modes are not merely nomenclature: when the target is a carrier with
time/frequency envelope or mixed degree-2 modulation, the matching full
spectral-jet space recovers it while J0, DCT, and coordinate-shuffled controls
do not.
```

Artifacts:

- `tf_jet_results.csv`
- `tf_jet_aggregate.csv`
- `tf_jet_family_aggregate.csv`
- `positive_margins.csv`
- `ridge_path.csv`
- `tf_jet_matrix.pdf`
- `high_order_margins.pdf`
