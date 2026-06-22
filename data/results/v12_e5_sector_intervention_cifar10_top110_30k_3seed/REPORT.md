# V12 Sector Intervention Report

Environment:

- Device: cuda
- State: final
- Dataset: cifar10
- Task: reconstruction
- Checkpoints: 3
- Rows: 24

## Aggregate

| mode | sector | n | features | score mean | score std | delta vs full | gain vs zero | coeff energy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ablate | axial_J0 | 3 | 56 | 0.4308 | 0.1006 | -0.3214 | 0.2131 | 0.5538 |
| ablate | const | 3 | 1 | 0.7523 | 0.0445 | 0.0001 | 0.5346 | 0.0065 |
| ablate | oblique_J0 | 3 | 52 | 0.4597 | 0.0897 | -0.2925 | 0.2420 | 0.4398 |
| full | all_positional | 3 | 109 | 0.7522 | 0.0447 | 0.0000 | 0.5345 | 1.0000 |
| keep | axial_J0 | 3 | 56 | 0.4591 | 0.0898 | -0.2931 | 0.2414 | 0.5538 |
| keep | const | 3 | 1 | 0.2177 | 0.0034 | -0.5345 | 0.0001 | 0.0065 |
| keep | oblique_J0 | 3 | 52 | 0.4307 | 0.1001 | -0.3215 | 0.2130 | 0.4398 |
| zero_bias | none | 3 | 0 | 0.2177 | 0.0035 | -0.5345 | 0.0000 | 0.0000 |

Notes:

- `keep` evaluates only the selected positional-bias sector.
- `ablate` evaluates the full positional bias with that sector removed.
- Coeff energy share is measured on learned positional coefficients, not on task loss.
