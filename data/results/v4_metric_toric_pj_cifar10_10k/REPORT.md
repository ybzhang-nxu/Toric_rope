# V4-C Metric Toric PJ Student Report

Environment:

- Device: cuda
- Dataset: cifar10
- Teacher bias: `results/v4_cifar10_bias_export_10k/bias_exports/cifar10_reconstruction_relative_2d_table_seed477_steps10000/bias_tables.npz`
- Patch size / grid side: 4 / 8
- Depth / dim / heads: 6 / 384 / 8
- Steps: 10000
- Teacher init: True

## Aggregate

| basis | n | score mean | best mean | final mean | train score | features | init R2 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| axis_plus_toric_residual_R0_top55 | 3 | 0.8599 | 0.8599 | 0.8592 | 0.8609 | 55 | 0.7843 |
| axis_plus_toric_residual_R2_top55 | 1 | 0.8229 | 0.8229 | 0.8225 | 0.8295 | 47 | 0.7578 |
| dct_top33 | 3 | 0.8468 | 0.8468 | 0.8468 | 0.8482 | 34 | 0.8039 |
| dct_top55 | 1 | 0.8275 | 0.8275 | 0.8279 | 0.8285 | 56 | 0.8649 |
| table_informed_toric_PJ_R0_top110 | 3 | 0.8662 | 0.8662 | 0.8673 | 0.8667 | 109 | 0.8030 |
| table_informed_toric_PJ_R0_top55 | 1 | 0.8133 | 0.8133 | 0.8138 | 0.8147 | 55 | 0.7374 |

Artifacts:

- `student_results.csv`
- `student_aggregate.csv`
- `student_curves.csv`
- `teacher_init_fits.csv`
- `basis_accuracy_boxplot.png`
- `train_test_curves.png`
