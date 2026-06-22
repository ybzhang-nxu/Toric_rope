# V8 TinyImageNet Mechanism Summary

Date: 2026-06-11

## Scope

```text
dataset=tiny-imagenet
task=reconstruction
checkpoint state=final
seed=626
eval_seed=3626
radius=4 unless noted
```

Compared checkpoints:

```text
Toric/PJ:
results/v8_tinyimagenet_toric_pj_top220_10k_3seed_from_relative10k_s426/checkpoints/tiny-imagenet_reconstruction_table_informed_toric_PJ_R0_top220_seed626_holdoutR4_steps10000_h1_b10_cl20_tail0_g0_constant_s0_r0.pt

Mixed:
results/v8_tinyimagenet_mixed_residual_dct32_trainradial_r4_10k_3seed_from_relative10k_s426/checkpoints/tiny-imagenet_reconstruction_mixed_toric_PJ_R0_top220_residual_dct_top32_seed626_holdoutR4_steps10000_h1_b10_cl20_tail0_g0_constant_s0_r0_trainr4.pt
```

## Layer Pool

| model | full | zero | condition | score | effect |
|---|---:|---:|---|---:|---:|
| Toric/PJ | 0.8279 | 0.2790 | L0 ablate | 0.6895 | drop +0.1384 |
| Toric/PJ | 0.8279 | 0.2790 | L0 keep | 0.7524 | gap +0.0755 |
| Toric/PJ | 0.8279 | 0.2790 | L0+L1 ablate | 0.6432 | drop +0.1847 |
| Toric/PJ | 0.8279 | 0.2790 | L0+L1 keep | 0.7977 | gap +0.0301 |
| Toric/PJ | 0.8279 | 0.2790 | L0-L3 ablate | 0.3037 | drop +0.5241 |
| Toric/PJ | 0.8279 | 0.2790 | L0-L3 keep | 0.8475 | gap -0.0197 |
| Mixed r4 | 0.8521 | 0.2484 | L1 ablate | 0.7535 | drop +0.0986 |
| Mixed r4 | 0.8521 | 0.2484 | L1 keep | 0.7267 | gap +0.1254 |
| Mixed r4 | 0.8521 | 0.2484 | L0+L1 ablate | 0.6616 | drop +0.1904 |
| Mixed r4 | 0.8521 | 0.2484 | L0+L1 keep | 0.8175 | gap +0.0346 |
| Mixed r4 | 0.8521 | 0.2484 | L0-L3 ablate | 0.2698 | drop +0.5823 |
| Mixed r4 | 0.8521 | 0.2484 | L0-L3 keep | 0.8665 | gap -0.0144 |

Readout:

```text
Both TinyImageNet compact mechanisms have a strong early local pool.
L0-L3 keep is sufficient and slightly exceeds full for both models.
L0-L3 ablate collapses almost to zero-bias:
Toric/PJ gain vs zero after ablate is +0.0248;
Mixed gain vs zero after ablate is +0.0214.
```

## Head Pool

| model | condition | score | effect |
|---|---|---:|---:|
| Toric/PJ | L0H4 keep | 0.5856 | gap +0.2423 |
| Toric/PJ | L0H6 keep | 0.6044 | gap +0.2235 |
| Toric/PJ | L0H4 ablate | 0.7415 | drop +0.0863 |
| Toric/PJ | L0H6 ablate | 0.7824 | drop +0.0455 |
| Toric/PJ | L0H4+H6 keep | 0.7101 | gap +0.1178 |
| Toric/PJ | L0H4+H6 ablate | 0.6639 | drop +0.1640 |
| Mixed r4 | L1H2 keep | 0.4971 | gap +0.3550 |
| Mixed r4 | L1H7 keep | 0.4284 | gap +0.4237 |
| Mixed r4 | L1H0 keep | 0.4008 | gap +0.4513 |
| Mixed r4 | L1H2 ablate | 0.8143 | drop +0.0378 |
| Mixed r4 | L1H7 ablate | 0.8256 | drop +0.0264 |
| Mixed r4 | L1H0 ablate | 0.8279 | drop +0.0242 |
| Mixed r4 | L1H2+H7+H0 keep | 0.7127 | gap +0.1394 |
| Mixed r4 | L1H2+H7+H0 ablate | 0.7600 | drop +0.0921 |

Readout:

```text
Single heads are not sufficient on TinyImageNet.
Toric/PJ has a clear L0H4/L0H6 head pool.
Mixed shifts the strongest head-pool signal to L1H2/L1H7/L1H0.
The selected top-head subsets are meaningful but weaker than the full L0-L3 pool, so the mechanism is distributed across early layers.
```

## Outputs

```text
results/v8_component_eval_tinyimagenet_toric_pj_top220_seed626_layer_l0_l3_10k/
results/v8_component_eval_tinyimagenet_mixed_residual_dct32_trainradial_r4_seed626_layer_l0_l3_10k/
results/v8_component_eval_tinyimagenet_toric_pj_top220_seed626_l0_l3_pool_10k/
results/v8_component_eval_tinyimagenet_mixed_residual_dct32_trainradial_r4_seed626_l0_l3_pool_10k/
results/v8_component_eval_tinyimagenet_toric_pj_top220_seed626_l0_head_scan_10k/
results/v8_component_eval_tinyimagenet_mixed_residual_dct32_trainradial_r4_seed626_l1_head_scan_10k/
results/v8_component_eval_tinyimagenet_toric_pj_top220_seed626_top_heads_10k/
results/v8_component_eval_tinyimagenet_mixed_residual_dct32_trainradial_r4_seed626_top_heads_10k/
```
