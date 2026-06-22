# V11-E CIFAR10 Ordinary Classification Medium Run

This is the medium-scale follow-up to the small V11-E CIFAR10 ordinary
classification smoke. It uses the existing `v3_real_vision_scaling.py`
classification path, not a true V11 route/connection classifier.

## Config

```text
dataset = cifar10
task = classification
mode = main
steps = 5000
seeds = 3
train_limit = 20000
test_limit = 5000
patch_size = 4
grid_side = 8
depth = 4
dim = 256
n_heads = 8
batch_size = 1024
amp = bf16
score_mode = best
lr_schedule = cosine_hold
lr_decay_steps = 2500
wall_sec = 3424.24
peak_cuda_memory = 3.76 GB
```

## Accuracy

| basis | score mean | score std | final mean | train score |
|---|---:|---:|---:|---:|
| axis_additive | 0.5811 | 0.0043 | 0.5805 | 1.0000 |
| no_pos_constant | 0.5754 | 0.0086 | 0.5745 | 1.0000 |
| relative_2d_table | 0.5741 | 0.0055 | 0.5733 | 1.0000 |
| toric_PJ_R2_coord_shuffle | 0.5715 | 0.0081 | 0.5707 | 1.0000 |
| toric_order0 | 0.5699 | 0.0040 | 0.5687 | 1.0000 |
| toric_PJ_R2 | 0.5683 | 0.0026 | 0.5674 | 1.0000 |

The medium run is a conservative/negative ordinary-classification result for
Toric-PJ. The best model is `axis_additive`. `toric_PJ_R2` is below
`no_pos_constant` by about 0.71 points and below `axis_additive` by about
1.27 points.

## Geometry And Bias Utility

Centered/interior geometry and bias ablation aggregates:

| basis | obl ratio | mixed ratio | diagonal ratio | top spectral mass | zero-bias delta | mean abs ablation delta |
|---|---:|---:|---:|---:|---:|---:|
| axis_additive | 0.0000 | 0.0000 | 0.0000 | 0.8982 | 0.00020 | 0.00043 |
| no_pos_constant | 0.0000 | 0.0000 | 0.0003 | 0.0003 | -0.00067 | 0.00040 |
| relative_2d_table | 0.3993 | 0.0193 | 0.3993 | 0.4577 | -0.00000 | 0.00063 |
| toric_PJ_R2 | 0.8696 | 0.1337 | 0.8696 | 0.4979 | 0.00107 | 0.00082 |
| toric_PJ_R2_coord_shuffle | 0.8703 | 0.0149 | 0.8703 | 0.0984 | -0.00087 | 0.00076 |
| toric_order0 | 0.9484 | 0.1557 | 0.9484 | 0.6538 | 0.00053 | 0.00041 |

Toric-PJ_R2 does learn strong oblique/mixed bias geometry, but the classifier
barely uses the bias at evaluation time. Zeroing or decomposing the bias changes
accuracy by roughly 0.1 point or less. This matches the accuracy result:
ordinary CIFAR10 classification does not currently provide strong support for
the Toric-PJ connection mechanism.

## Takeaway

Supported:

```text
Medium-scale ordinary CIFAR10 classification runs successfully on the AD6000.
The model overfits train=1.0 and reaches about 57-58% test accuracy.
Axis-additive relative position gives the best small improvement over no-position.
Toric-PJ_R2 still learns visibly oblique/mixed tables.
```

Not supported:

```text
Toric-PJ_R2 ordinary CIFAR10 downstream gain.
Full vision downstream success from ordinary image classification.
Evidence that ordinary CIFAR10 class labels require the V11 route/connection mechanism.
```

Next implication:

```text
Do not spend the next round merely scaling this same ordinary-classification
setup. The stronger V11 path is a true route/connection classifier trained on
image labels, with intervention/drop and holonomy-aware ablations.
```

