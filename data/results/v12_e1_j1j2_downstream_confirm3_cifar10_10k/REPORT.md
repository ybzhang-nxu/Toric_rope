# V4-C Metric Toric PJ Student Report

Environment:

- Device: cuda
- Dataset: cifar10
- Teacher bias: `MetricToric/results/v4_cifar10_bias_export_10k/bias_exports/cifar10_reconstruction_relative_2d_table_seed477_steps10000/bias_tables.npz`
- Patch size / grid side: 4 / 8
- Depth / dim / heads: 6 / 384 / 8
- Steps: 10000
- Teacher init: True

## Aggregate

| basis | n | score mean | best mean | final mean | train score | features | init R2 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| v12_matched_axis_residual_toric_full_J1_atoms108 | 3 | 0.8610 | 0.8610 | 0.8609 | 0.8616 | 107 | 0.8155 |
| v12_matched_axis_residual_toric_full_J2_atoms108 | 3 | 0.8575 | 0.8575 | 0.8578 | 0.8587 | 101 | 0.7991 |

Artifacts:

- `student_results.csv`
- `student_aggregate.csv`
- `student_curves.csv`
- `teacher_init_fits.csv`
- `basis_accuracy_boxplot.png`
- `train_test_curves.png`
