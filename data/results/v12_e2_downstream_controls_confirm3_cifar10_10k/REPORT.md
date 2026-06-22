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
| v12_freqsrc_axis_J0_atoms108 | 3 | 0.8651 | 0.8651 | 0.8648 | 0.8664 | 107 | 0.7168 |
| v12_freqsrc_random_J0_atoms108 | 3 | 0.8473 | 0.8473 | 0.8473 | 0.8535 | 107 | 0.8028 |
| v12_freqsrc_table_shuffled_J0_atoms108 | 3 | 0.8666 | 0.8666 | 0.8671 | 0.8683 | 107 | 0.8030 |

Artifacts:

- `student_results.csv`
- `student_aggregate.csv`
- `student_curves.csv`
- `teacher_init_fits.csv`
- `basis_accuracy_boxplot.png`
- `train_test_curves.png`
