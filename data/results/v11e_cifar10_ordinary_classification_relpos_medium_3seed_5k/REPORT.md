# V3-D Real Vision Scaling Report

Environment:

- Device: cuda
- Dataset: cifar10
- Mode: main
- Patch size / grid side: 4 / 8
- Depth / dim / heads: 4 / 256 / 8
- Steps: 5000
- Seeds: 3
- Score mode: best
- LR schedule: cosine_hold / decay steps 2500
- Wall seconds: 3424.24

## Aggregate

| dataset | task | basis | n | score mean | score std | best mean | final mean | train score | features | params |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | classification | axis_additive | 3 | 0.5811 | 0.0043 | 0.5811 | 0.5805 | 1.0000 | 5 | 2136282 |
| cifar10 | classification | no_pos_constant | 3 | 0.5754 | 0.0086 | 0.5754 | 0.5745 | 1.0000 | 1 | 2136154 |
| cifar10 | classification | relative_2d_table | 3 | 0.5741 | 0.0055 | 0.5741 | 0.5733 | 1.0000 | 225 | 2143322 |
| cifar10 | classification | toric_PJ_R2 | 3 | 0.5683 | 0.0026 | 0.5683 | 0.5674 | 1.0000 | 55 | 2137882 |
| cifar10 | classification | toric_PJ_R2_coord_shuffle | 3 | 0.5715 | 0.0081 | 0.5715 | 0.5707 | 1.0000 | 55 | 2137882 |
| cifar10 | classification | toric_order0 | 3 | 0.5699 | 0.0040 | 0.5699 | 0.5687 | 1.0000 | 7 | 2136346 |

Notes:

- Classification scores are accuracy.
- Reconstruction scores are masked-patch R2.
- `score mean` follows the selected score mode; `best mean` and `final mean` are shown separately when available.
- `relative_2d_table` is the high-capacity upper-bound style baseline.

Artifacts:

- `real_vision_results.csv`
- `real_vision_aggregate.csv`
- `real_vision_curves.csv`
- `basis_accuracy_boxplot.png`
- `train_test_curves.png`
