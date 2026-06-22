# V18 NSynth Learned-Bias Holdout Projection

Heldout-offset diagnostics for learned NSynth/CQT scalar relative-bias tables.
For each split, frequency centers and DCT atoms are selected from the training
offset window only; coefficients are fit on training offsets and scored on heldout offsets.

## Heldout Scores At Default Ridge

Default ridge: `1e-06`.

| split | basis | n | heldout R2 mean | std | min | max |
|---|---|---:|---:|---:|---:|---:|
| random | fft_top18_J2 | 45 | 0.9618 | 0.0145 | 0.8930 | 0.9780 |
| random | fft_top18_J1 | 45 | 0.9436 | 0.0146 | 0.8976 | 0.9668 |
| random | dct_top73 | 45 | 0.9260 | 0.0206 | 0.8600 | 0.9591 |
| random | fft_top18_J0 | 45 | 0.8690 | 0.0328 | 0.7764 | 0.9140 |
| random | fft_top18_J2_coord_shuffle | 45 | -0.1893 | 0.0299 | -0.2608 | -0.1293 |

## Heldout Margins

| split | ridge | comparison | n | heldout mean | heldout min | positive |
|---|---:|---|---:|---:|---:|---:|
| random | 1e-06 | J1_minus_J0 | 45 | 0.0746 | 0.0388 | 45 |
| random | 1e-06 | J2_minus_J1 | 45 | 0.0183 | -0.0228 | 43 |
| random | 1e-06 | J2_minus_shuffle | 45 | 1.1511 | 1.0762 | 45 |

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
