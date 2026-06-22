# V4-E Offset Holdout Report

Environment:

- Device: cuda
- Dataset: cifar10
- Teacher bias: `results/v4_cifar10_bias_export_10k/bias_exports/cifar10_reconstruction_relative_2d_table_seed477_steps10000/bias_tables.npz`
- Teacher-init ridge: 0.0001
- Holdout radius: 4
- Steps: 10000
- Seeds: 1

## Aggregate

| basis | n | features | score mean | score std | final full | final visible-only | full-visible gain | heldout bias rms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| axis_plus_toric_residual_R0_top55 | 1 | 55 | 0.6474 | 0.0000 | 0.6476 | 0.8150 | -0.1674 | 0.0506 |
| dct_top33 | 1 | 34 | -0.0955 | 0.0000 | -0.0930 | 0.7849 | -0.8779 | 0.3434 |
| relative_2d_table | 1 | 225 | 0.8439 | 0.0000 | 0.8436 | 0.8436 | 0.0001 | 0.0000 |
| table_informed_toric_PJ_R0_top110 | 1 | 109 | 0.5674 | 0.0000 | 0.5670 | 0.8539 | -0.2869 | 0.1919 |

Notes:

- Training keeps the attention graph unchanged but masks positional-bias contributions for held-out offsets.
- `final full` evaluates the learned compact function on all offsets.
- `final visible-only` keeps held-out offset bias at zero during eval.
- Positive full-visible gain means the learned functional bias uses extrapolated held-out offsets beneficially.
