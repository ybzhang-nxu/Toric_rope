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
| v12_freqsrc_fixed_J0_atoms108 | 3 | 0.8757 | 0.8757 | 0.8763 | 0.8790 | 107 | 0.7843 |
| v12_freqsrc_table_J0_atoms108 | 3 | 0.8665 | 0.8665 | 0.8663 | 0.8677 | 107 | 0.8030 |
| v12_matched_axis_residual_toric_full_J0_atoms108 | 3 | 0.8634 | 0.8634 | 0.8636 | 0.8654 | 107 | 0.8397 |

Artifacts:

- `student_results.csv`
- `student_aggregate.csv`
- `student_curves.csv`
- `teacher_init_fits.csv`
- `basis_accuracy_boxplot.png`
- `train_test_curves.png`
