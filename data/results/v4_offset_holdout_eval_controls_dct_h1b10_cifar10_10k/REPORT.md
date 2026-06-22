# V4-E Offset Holdout Report

Environment:

- Device: cuda
- Dataset: cifar10
- Teacher bias: `results/v4_cifar10_bias_export_10k/bias_exports/cifar10_reconstruction_relative_2d_table_seed477_steps10000/bias_tables.npz`
- Teacher-init ridge: 0.0001
- Heldout bias MSE weight: 1.0
- Boundary smoothness weight: 10.0
- Regularization schedule: constant
- Regularization start/ramp: 0 / 0
- Eval control modes: heldout_clamp,radial_decay
- Eval radial decay gammas: 0.25,0.5,1,2
- Holdout radius: 4
- Steps: 10000
- Seeds: 3

## Aggregate

| basis | n | features | R | h-wt | b-wt | reg | start | ramp | score mean | score std | final full | final visible-only | full-visible gain | heldout bias rms |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dct_top33 | 3 | 34 | 4 | 1 | 10 | constant | 0 | 0 | 0.6707 | 0.0245 | 0.6705 | 0.7824 | -0.1119 | 0.2856 |

Notes:

- Training keeps the attention graph unchanged but masks positional-bias contributions for held-out offsets.
- `final full` evaluates the learned compact function on all offsets.
- `final visible-only` keeps held-out offset bias at zero during eval.
- Positive full-visible gain means the learned functional bias uses extrapolated held-out offsets beneficially.
- Extra final-time controls, when enabled, are written to `offset_holdout_eval_controls.csv`.
