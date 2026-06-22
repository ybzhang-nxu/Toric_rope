# V18 NSynth Learned-Bias Holdout Projection

Heldout-offset diagnostics for learned NSynth/CQT scalar relative-bias tables.
For each split, frequency centers and DCT atoms are selected from the training
offset window only; coefficients are fit on training offsets and scored on heldout offsets.

## Heldout Scores At Default Ridge

Default ridge: `1e-06`.

| split | basis | n | heldout R2 mean | std | min | max |
|---|---|---:|---:|---:|---:|---:|
| random | fft_top36_J1 | 45 | 0.9641 | 0.0137 | 0.9340 | 0.9817 |
| random | fft_top36_J2 | 45 | 0.9582 | 0.1086 | 0.2414 | 0.9908 |
| random | dct_top73 | 45 | 0.9260 | 0.0206 | 0.8600 | 0.9591 |
| random | fft_top36_J0 | 45 | 0.8954 | 0.0263 | 0.8390 | 0.9436 |
| random | fft_top36_J2_coord_shuffle | 45 | -0.3897 | 0.0840 | -0.6708 | -0.2605 |

## Heldout Margins

| split | ridge | comparison | n | heldout mean | heldout min | positive |
|---|---:|---|---:|---:|---:|---:|
| random | 1e-06 | J1_minus_J0 | 45 | 0.0686 | 0.0318 | 45 |
| random | 1e-06 | J2_minus_J1 | 45 | -0.0058 | -0.7216 | 39 |
| random | 1e-06 | J2_minus_shuffle | 45 | 1.3480 | 0.7091 | 45 |

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
