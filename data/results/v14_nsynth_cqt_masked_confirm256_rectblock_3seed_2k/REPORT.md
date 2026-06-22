# V14 NSynth CQT Masked Reconstruction

Status: ok
Dataset root: `data/nsynth/nsynth-test`
Records: 256 (205 train / 51 test)
CQT: 128x84, patch 4x6
Token grid: 32x14
Masking: rect_block at rate 0.25

## Aggregate

| basis | features | n | masked R2 | onset R2 | middle R2 | decay R2 |
|---|---:|---:|---:|---:|---:|---:|
| axis_additive | 5 | 3 | -0.0258 +/- 0.0203 | 0.0013 | 0.0464 | -7.4306 |
| toric_j1 | 41 | 3 | -0.0276 +/- 0.0203 | -0.0115 | 0.0536 | -8.7558 |
| toric_j2_coord_shuffle | 73 | 3 | -0.0299 +/- 0.0598 | -0.0087 | 0.0388 | -8.7613 |
| toric_j0 | 9 | 3 | -0.0321 +/- 0.0195 | -0.0030 | 0.0423 | -7.4475 |
| toric_j2 | 73 | 3 | -0.0421 +/- 0.0265 | 0.0060 | 0.0308 | -9.3476 |
| no_pos_constant | 1 | 3 | -0.0421 +/- 0.0237 | -0.0080 | 0.0423 | -9.3890 |
| dct_lowfreq33 | 33 | 3 | -0.0651 +/- 0.0236 | -0.0225 | 0.0211 | -9.5619 |

## Reading

This is the first real-data gate for the time-frequency higher-order jet story.
It should be treated as a smoke/pilot unless run with enough NSynth examples,
multiple seeds, and longer training.

Artifacts:

- `nsynth_cqt_results.csv`
- `nsynth_cqt_aggregate.csv`
- `nsynth_cqt_summary.json`
- `nsynth_cqt_masked_r2.pdf`
