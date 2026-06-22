# V4-E Offset Holdout Report

Environment:

- Device: cuda
- Dataset: cifar10
- Teacher bias: `results/v4_cifar10_bias_export_10k/bias_exports/cifar10_reconstruction_relative_2d_table_seed477_steps10000/bias_tables.npz`
- Teacher-init ridge: 0.0001
- Heldout bias MSE weight: 10.0
- Boundary smoothness weight: 1.0
- Regularization schedule: constant
- Regularization start/ramp: 0 / 0
- Holdout radius: 4
- Steps: 10000
- Seeds: 3

## Aggregate

| basis | n | features | h-wt | b-wt | reg | start | ramp | score mean | score std | final full | final visible-only | full-visible gain | heldout bias rms |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| axis_plus_toric_residual_R0_top55 | 3 | 55 | 10 | 1 | constant | 0 | 0 | 0.7659 | 0.0412 | 0.7661 | 0.8075 | -0.0414 | 0.0196 |

Notes:

- Training keeps the attention graph unchanged but masks positional-bias contributions for held-out offsets.
- `final full` evaluates the learned compact function on all offsets.
- `final visible-only` keeps held-out offset bias at zero during eval.
- Positive full-visible gain means the learned functional bias uses extrapolated held-out offsets beneficially.
