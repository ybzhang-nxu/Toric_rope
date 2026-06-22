# V3-D Real Vision Scaling Report

Environment:

- Device: cuda
- Dataset: cifar10
- Mode: overnight
- Patch size / grid side: 4 / 8
- Depth / dim / heads: 6 / 384 / 8
- Steps: 10000
- Seeds: 1
- Score mode: best
- LR schedule: cosine_hold / decay steps 5000
- Wall seconds: 4546.67

## Aggregate

| dataset | task | basis | n | score mean | score std | best mean | final mean | train score | features | params |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | reconstruction | axis_additive | 1 | 0.6042 | 0.0000 | 0.6042 | 0.6035 | 0.6100 | 5 | 7145002 |
| cifar10 | reconstruction | pruned_toric_PJ | 1 | 0.6765 | 0.0000 | 0.6765 | 0.6775 | 0.6807 | 33 | 7146346 |
| cifar10 | reconstruction | relative_2d_table | 1 | 0.8362 | 0.0000 | 0.8362 | 0.8377 | 0.8367 | 225 | 7155562 |
| cifar10 | reconstruction | toric_PJ_R2 | 1 | 0.7011 | 0.0000 | 0.7011 | 0.7004 | 0.7053 | 55 | 7147402 |
| cifar10 | reconstruction | toric_PJ_R2_coord_shuffle | 1 | 0.3927 | 0.0000 | 0.3927 | 0.3919 | 0.4087 | 55 | 7147402 |

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
