# V18 NSynth Learned-Bias Holdout Projection

Heldout-offset diagnostics for learned NSynth/CQT scalar relative-bias tables.
For each split, frequency centers and DCT atoms are selected from the training
offset window only; coefficients are fit on training offsets and scored on heldout offsets.

## Heldout Scores At Default Ridge

Default ridge: `1e-06`.

| split | basis | n | heldout R2 mean | std | min | max |
|---|---|---:|---:|---:|---:|---:|
| checkerboard | fft_top6_J0 | 15 | 0.2405 | 0.2848 | -0.2869 | 0.6204 |
| checkerboard | fft_top6_J2_coord_shuffle | 15 | -0.0854 | 0.0114 | -0.1037 | -0.0688 |
| checkerboard | dct_top73 | 15 | -85.0553 | 86.4525 | -304.8092 | 0.0425 |
| checkerboard | fft_top6_J1 | 15 | -217.1021 | 161.1926 | -470.1086 | -3.0557 |
| checkerboard | fft_top6_J2 | 15 | -3222.5986 | 2157.7723 | -7803.7241 | -313.8757 |
| outer_shell | fft_top6_J0 | 15 | 0.2550 | 0.1962 | -0.1413 | 0.6197 |
| outer_shell | fft_top6_J2_coord_shuffle | 15 | -0.8786 | 0.2539 | -1.2831 | -0.4376 |
| outer_shell | dct_top73 | 15 | -9.1047 | 23.7479 | -94.4661 | -0.1062 |
| outer_shell | fft_top6_J1 | 15 | -10.6235 | 6.1487 | -22.4040 | -2.5653 |
| outer_shell | fft_top6_J2 | 15 | -214.8352 | 126.9829 | -510.0080 | -12.2245 |
| random | dct_top73 | 45 | 0.9243 | 0.0252 | 0.8485 | 0.9582 |
| random | fft_top6_J2 | 45 | 0.9085 | 0.0300 | 0.8109 | 0.9504 |
| random | fft_top6_J1 | 45 | 0.8644 | 0.0385 | 0.7552 | 0.9244 |
| random | fft_top6_J0 | 45 | 0.7909 | 0.0464 | 0.6682 | 0.8723 |
| random | fft_top6_J2_coord_shuffle | 45 | -0.0505 | 0.0170 | -0.0899 | -0.0130 |

## Heldout Margins

| split | ridge | comparison | n | heldout mean | heldout min | positive |
|---|---:|---|---:|---:|---:|---:|
| checkerboard | 1e-08 | J1_minus_J0 | 15 | -223.1670 | -487.7542 | 0 |
| checkerboard | 1e-08 | J2_minus_J1 | 15 | -4686465.7354 | -68624683.8069 | 0 |
| checkerboard | 1e-08 | J2_minus_shuffle | 15 | -4686688.5744 | -68624863.9102 | 0 |
| checkerboard | 1e-06 | J1_minus_J0 | 15 | -217.3427 | -470.0856 | 0 |
| checkerboard | 1e-06 | J2_minus_J1 | 15 | -3005.4965 | -7541.0768 | 0 |
| checkerboard | 1e-06 | J2_minus_shuffle | 15 | -3222.5132 | -7803.6482 | 0 |
| checkerboard | 1e-04 | J1_minus_J0 | 15 | -62.5848 | -174.4619 | 0 |
| checkerboard | 1e-04 | J2_minus_J1 | 15 | 37.4532 | -33.1301 | 10 |
| checkerboard | 1e-04 | J2_minus_shuffle | 15 | -24.8125 | -41.4818 | 0 |
| outer_shell | 1e-08 | J1_minus_J0 | 15 | -14.1382 | -32.6020 | 0 |
| outer_shell | 1e-08 | J2_minus_J1 | 15 | -2024279.3938 | -11436696.1796 | 0 |
| outer_shell | 1e-08 | J2_minus_shuffle | 15 | -2024292.3895 | -11436719.8760 | 0 |
| outer_shell | 1e-06 | J1_minus_J0 | 15 | -10.8784 | -22.6091 | 0 |
| outer_shell | 1e-06 | J2_minus_J1 | 15 | -204.2117 | -495.7108 | 0 |
| outer_shell | 1e-06 | J2_minus_shuffle | 15 | -213.9566 | -509.4091 | 0 |
| outer_shell | 1e-04 | J1_minus_J0 | 15 | -4.5185 | -10.1755 | 0 |
| outer_shell | 1e-04 | J2_minus_J1 | 15 | -4.3078 | -12.1256 | 2 |
| outer_shell | 1e-04 | J2_minus_shuffle | 15 | -7.7156 | -15.5219 | 0 |
| random | 1e-08 | J1_minus_J0 | 45 | 0.0734 | 0.0240 | 45 |
| random | 1e-08 | J2_minus_J1 | 45 | -0.6216 | -23.1723 | 39 |
| random | 1e-08 | J2_minus_shuffle | 45 | 0.3044 | -22.2276 | 42 |
| random | 1e-06 | J1_minus_J0 | 45 | 0.0734 | 0.0240 | 45 |
| random | 1e-06 | J2_minus_J1 | 45 | 0.0442 | 0.0123 | 45 |
| random | 1e-06 | J2_minus_shuffle | 45 | 0.9590 | 0.8686 | 45 |
| random | 1e-04 | J1_minus_J0 | 45 | 0.0737 | 0.0238 | 45 |
| random | 1e-04 | J2_minus_J1 | 45 | 0.0388 | 0.0107 | 45 |
| random | 1e-04 | J2_minus_shuffle | 45 | 0.9473 | 0.8608 | 45 |

## Reading

A stable positive heldout J2-J1 margin would strengthen the V17 result by
showing that the high-order jet contribution is not only an in-window
projection artifact.  Random and checkerboard splits test interpolation;
outer-shell tests a harder boundary/extrapolation setting.

Artifacts:

- `learned_bias_holdout_rows.csv`
- `learned_bias_holdout_margins.csv`
- `learned_bias_holdout_basis_aggregate.csv`
- `learned_bias_holdout_margin_aggregate.csv`
- `learned_bias_holdout_margins.pdf`
- `summary.json`
