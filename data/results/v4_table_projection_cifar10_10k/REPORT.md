# V4 Table Projection Report

Teacher:

- dataset: cifar10
- task: reconstruction
- basis: relative_2d_table
- seed: 477
- steps: 10000
- best_step: 9000
- score: 0.8361638188362122
- final_score: 0.8376511335372925
- train_score: 0.8367145657539368
- grid_side: 8
- n_layers: 6
- n_heads: 8
- num_features: 225

## Aggregate

| variant | target | budget | features | n | table R2 | residual R2 |
|---|---|---:|---:|---:|---:|---:|
| axis_full_29 | axis_plus_residual | 5 | 29 | 48 | 0.7666 | 0.0000 |
| axis_full_29 | axis_plus_residual | 15 | 29 | 48 | 0.7666 | 0.0000 |
| axis_full_29 | axis_plus_residual | 33 | 29 | 48 | 0.7666 | 0.0000 |
| axis_full_29 | axis_plus_residual | 55 | 29 | 48 | 0.7666 | 0.0000 |
| axis_full_29 | axis_plus_residual | 110 | 29 | 48 | 0.7666 | 0.0000 |
| axis_full_29 | axis_plus_residual | 225 | 29 | 48 | 0.7666 | 0.0000 |
| axis_full_29 | full_table | 5 | 29 | 48 | 0.7666 | nan |
| axis_full_29 | full_table | 15 | 29 | 48 | 0.7666 | nan |
| axis_full_29 | full_table | 33 | 29 | 48 | 0.7666 | nan |
| axis_full_29 | full_table | 55 | 29 | 48 | 0.7666 | nan |
| axis_full_29 | full_table | 110 | 29 | 48 | 0.7666 | nan |
| axis_full_29 | full_table | 225 | 29 | 48 | 0.7666 | nan |
| axis_full_29 | oblique_residual | 5 | 29 | 48 | -0.7141 | 0.0000 |
| axis_full_29 | oblique_residual | 15 | 29 | 48 | -0.7141 | 0.0000 |
| axis_full_29 | oblique_residual | 33 | 29 | 48 | -0.7141 | 0.0000 |
| axis_full_29 | oblique_residual | 55 | 29 | 48 | -0.7141 | 0.0000 |
| axis_full_29 | oblique_residual | 110 | 29 | 48 | -0.7141 | 0.0000 |
| axis_full_29 | oblique_residual | 225 | 29 | 48 | -0.7141 | 0.0000 |
| axis_lowrank_5 | axis_plus_residual | 5 | 5 | 48 | 0.7666 | -0.0000 |
| axis_lowrank_5 | axis_plus_residual | 15 | 5 | 48 | 0.7666 | -0.0000 |
| axis_lowrank_5 | axis_plus_residual | 33 | 5 | 48 | 0.7666 | -0.0000 |
| axis_lowrank_5 | axis_plus_residual | 55 | 5 | 48 | 0.7666 | -0.0000 |
| axis_lowrank_5 | axis_plus_residual | 110 | 5 | 48 | 0.7666 | -0.0000 |
| axis_lowrank_5 | axis_plus_residual | 225 | 5 | 48 | 0.7666 | -0.0000 |
| axis_lowrank_5 | full_table | 5 | 5 | 48 | 0.1613 | nan |
| axis_lowrank_5 | full_table | 15 | 5 | 48 | 0.1613 | nan |
| axis_lowrank_5 | full_table | 33 | 5 | 48 | 0.1613 | nan |
| axis_lowrank_5 | full_table | 55 | 5 | 48 | 0.1613 | nan |
| axis_lowrank_5 | full_table | 110 | 5 | 48 | 0.1613 | nan |
| axis_lowrank_5 | full_table | 225 | 5 | 48 | 0.1613 | nan |
| axis_lowrank_5 | oblique_residual | 5 | 5 | 48 | -0.7141 | -0.0000 |
| axis_lowrank_5 | oblique_residual | 15 | 5 | 48 | -0.7141 | -0.0000 |
| axis_lowrank_5 | oblique_residual | 33 | 5 | 48 | -0.7141 | -0.0000 |
| axis_lowrank_5 | oblique_residual | 55 | 5 | 48 | -0.7141 | -0.0000 |
| axis_lowrank_5 | oblique_residual | 110 | 5 | 48 | -0.7141 | -0.0000 |
| axis_lowrank_5 | oblique_residual | 225 | 5 | 48 | -0.7141 | -0.0000 |
| axis_plus_lc_toric_residual | axis_plus_residual | 5 | 19 | 48 | 0.8206 | 0.2490 |
| axis_plus_lc_toric_residual | axis_plus_residual | 15 | 19 | 48 | 0.8206 | 0.2490 |
| axis_plus_lc_toric_residual | axis_plus_residual | 33 | 19 | 48 | 0.8206 | 0.2490 |
| axis_plus_lc_toric_residual | axis_plus_residual | 55 | 19 | 48 | 0.8206 | 0.2490 |
| axis_plus_lc_toric_residual | axis_plus_residual | 110 | 73 | 48 | 0.8589 | 0.4168 |
| axis_plus_lc_toric_residual | axis_plus_residual | 225 | 181 | 48 | 0.9059 | 0.6203 |
| axis_plus_lc_toric_residual | full_table | 5 | 19 | 48 | 0.5800 | nan |
| axis_plus_lc_toric_residual | full_table | 15 | 19 | 48 | 0.5800 | nan |
| axis_plus_lc_toric_residual | full_table | 33 | 19 | 48 | 0.5800 | nan |
| axis_plus_lc_toric_residual | full_table | 55 | 19 | 48 | 0.5800 | nan |
| axis_plus_lc_toric_residual | full_table | 110 | 73 | 48 | 0.7125 | nan |
| axis_plus_lc_toric_residual | full_table | 225 | 181 | 48 | 0.8454 | nan |
| axis_plus_lc_toric_residual | oblique_residual | 5 | 19 | 48 | -0.6542 | 0.2490 |
| axis_plus_lc_toric_residual | oblique_residual | 15 | 19 | 48 | -0.6542 | 0.2490 |
| axis_plus_lc_toric_residual | oblique_residual | 33 | 19 | 48 | -0.6542 | 0.2490 |
| axis_plus_lc_toric_residual | oblique_residual | 55 | 19 | 48 | -0.6542 | 0.2490 |
| axis_plus_lc_toric_residual | oblique_residual | 110 | 73 | 48 | -0.6234 | 0.4168 |
| axis_plus_lc_toric_residual | oblique_residual | 225 | 181 | 48 | -0.5755 | 0.6203 |
| axis_plus_toric_residual_R0 | axis_plus_residual | 5 | 3 | 48 | 0.7937 | 0.1178 |
| axis_plus_toric_residual_R0 | axis_plus_residual | 15 | 3 | 48 | 0.7937 | 0.1178 |
| axis_plus_toric_residual_R0 | axis_plus_residual | 33 | 5 | 48 | 0.7937 | 0.1178 |
| axis_plus_toric_residual_R0 | axis_plus_residual | 55 | 27 | 48 | 0.8661 | 0.4283 |
| axis_plus_toric_residual_R0 | axis_plus_residual | 110 | 81 | 48 | 0.9209 | 0.6708 |
| axis_plus_toric_residual_R0 | axis_plus_residual | 225 | 197 | 48 | 0.9753 | 0.8984 |
| axis_plus_toric_residual_R0 | full_table | 5 | 3 | 48 | 0.4256 | nan |
| axis_plus_toric_residual_R0 | full_table | 15 | 3 | 48 | 0.4256 | nan |
| axis_plus_toric_residual_R0 | full_table | 33 | 5 | 48 | 0.4256 | nan |
| axis_plus_toric_residual_R0 | full_table | 55 | 27 | 48 | 0.7405 | nan |
| axis_plus_toric_residual_R0 | full_table | 110 | 81 | 48 | 0.8758 | nan |
| axis_plus_toric_residual_R0 | full_table | 225 | 197 | 48 | 0.9572 | nan |
| axis_plus_toric_residual_R0 | oblique_residual | 5 | 3 | 48 | -0.6871 | 0.1178 |
| axis_plus_toric_residual_R0 | oblique_residual | 15 | 3 | 48 | -0.6871 | 0.1178 |
| axis_plus_toric_residual_R0 | oblique_residual | 33 | 5 | 48 | -0.6871 | 0.1178 |
| axis_plus_toric_residual_R0 | oblique_residual | 55 | 27 | 48 | -0.6147 | 0.4283 |
| axis_plus_toric_residual_R0 | oblique_residual | 110 | 81 | 48 | -0.5599 | 0.6708 |
| axis_plus_toric_residual_R0 | oblique_residual | 225 | 197 | 48 | -0.5055 | 0.8984 |
| axis_plus_toric_residual_R1 | axis_plus_residual | 5 | 11 | 48 | 0.8050 | 0.1695 |
| axis_plus_toric_residual_R1 | axis_plus_residual | 15 | 11 | 48 | 0.8050 | 0.1695 |
| axis_plus_toric_residual_R1 | axis_plus_residual | 33 | 11 | 48 | 0.8050 | 0.1695 |
| axis_plus_toric_residual_R1 | axis_plus_residual | 55 | 21 | 48 | 0.8050 | 0.1695 |
| axis_plus_toric_residual_R1 | axis_plus_residual | 110 | 81 | 48 | 0.8657 | 0.4385 |
| axis_plus_toric_residual_R1 | axis_plus_residual | 225 | 191 | 48 | 0.9149 | 0.6626 |
| axis_plus_toric_residual_R1 | full_table | 5 | 11 | 48 | 0.5028 | nan |
| axis_plus_toric_residual_R1 | full_table | 15 | 11 | 48 | 0.5028 | nan |
| axis_plus_toric_residual_R1 | full_table | 33 | 11 | 48 | 0.5028 | nan |
| axis_plus_toric_residual_R1 | full_table | 55 | 21 | 48 | 0.5028 | nan |
| axis_plus_toric_residual_R1 | full_table | 110 | 81 | 48 | 0.7602 | nan |
| axis_plus_toric_residual_R1 | full_table | 225 | 191 | 48 | 0.8674 | nan |
| axis_plus_toric_residual_R1 | oblique_residual | 5 | 11 | 48 | -0.6836 | 0.1695 |
| axis_plus_toric_residual_R1 | oblique_residual | 15 | 11 | 48 | -0.6836 | 0.1695 |
| axis_plus_toric_residual_R1 | oblique_residual | 33 | 11 | 48 | -0.6836 | 0.1695 |
| axis_plus_toric_residual_R1 | oblique_residual | 55 | 21 | 48 | -0.6836 | 0.1695 |
| axis_plus_toric_residual_R1 | oblique_residual | 110 | 81 | 48 | -0.6220 | 0.4385 |
| axis_plus_toric_residual_R1 | oblique_residual | 225 | 191 | 48 | -0.5688 | 0.6626 |
| axis_plus_toric_residual_R2 | axis_plus_residual | 5 | 19 | 48 | 0.8206 | 0.2490 |
| axis_plus_toric_residual_R2 | axis_plus_residual | 15 | 19 | 48 | 0.8206 | 0.2490 |
| axis_plus_toric_residual_R2 | axis_plus_residual | 33 | 19 | 48 | 0.8206 | 0.2490 |
| axis_plus_toric_residual_R2 | axis_plus_residual | 55 | 19 | 48 | 0.8206 | 0.2490 |
| axis_plus_toric_residual_R2 | axis_plus_residual | 110 | 73 | 48 | 0.8589 | 0.4168 |
| axis_plus_toric_residual_R2 | axis_plus_residual | 225 | 181 | 48 | 0.9059 | 0.6203 |
| axis_plus_toric_residual_R2 | full_table | 5 | 19 | 48 | 0.5800 | nan |
| axis_plus_toric_residual_R2 | full_table | 15 | 19 | 48 | 0.5800 | nan |
| axis_plus_toric_residual_R2 | full_table | 33 | 19 | 48 | 0.5800 | nan |
| axis_plus_toric_residual_R2 | full_table | 55 | 19 | 48 | 0.5800 | nan |
| axis_plus_toric_residual_R2 | full_table | 110 | 73 | 48 | 0.7125 | nan |
| axis_plus_toric_residual_R2 | full_table | 225 | 181 | 48 | 0.8454 | nan |
| axis_plus_toric_residual_R2 | oblique_residual | 5 | 19 | 48 | -0.6542 | 0.2490 |
| axis_plus_toric_residual_R2 | oblique_residual | 15 | 19 | 48 | -0.6542 | 0.2490 |
| axis_plus_toric_residual_R2 | oblique_residual | 33 | 19 | 48 | -0.6542 | 0.2490 |
| axis_plus_toric_residual_R2 | oblique_residual | 55 | 19 | 48 | -0.6542 | 0.2490 |
| axis_plus_toric_residual_R2 | oblique_residual | 110 | 73 | 48 | -0.6234 | 0.4168 |
| axis_plus_toric_residual_R2 | oblique_residual | 225 | 181 | 48 | -0.5755 | 0.6203 |
| fixed_toric_PJ_R2 | axis_plus_residual | 5 | 55 | 48 | 0.8776 | 0.4934 |
| fixed_toric_PJ_R2 | axis_plus_residual | 15 | 55 | 48 | 0.8776 | 0.4934 |
| fixed_toric_PJ_R2 | axis_plus_residual | 33 | 55 | 48 | 0.8776 | 0.4934 |
| fixed_toric_PJ_R2 | axis_plus_residual | 55 | 55 | 48 | 0.8776 | 0.4934 |
| fixed_toric_PJ_R2 | axis_plus_residual | 110 | 55 | 48 | 0.8776 | 0.4934 |
| fixed_toric_PJ_R2 | axis_plus_residual | 225 | 55 | 48 | 0.8776 | 0.4934 |
| fixed_toric_PJ_R2 | full_table | 5 | 55 | 48 | 0.6541 | nan |
| fixed_toric_PJ_R2 | full_table | 15 | 55 | 48 | 0.6541 | nan |
| fixed_toric_PJ_R2 | full_table | 33 | 55 | 48 | 0.6541 | nan |
| fixed_toric_PJ_R2 | full_table | 55 | 55 | 48 | 0.6541 | nan |
| fixed_toric_PJ_R2 | full_table | 110 | 55 | 48 | 0.6541 | nan |
| fixed_toric_PJ_R2 | full_table | 225 | 55 | 48 | 0.6541 | nan |
| fixed_toric_PJ_R2 | oblique_residual | 5 | 55 | 48 | -0.5964 | 0.4934 |
| fixed_toric_PJ_R2 | oblique_residual | 15 | 55 | 48 | -0.5964 | 0.4934 |
| fixed_toric_PJ_R2 | oblique_residual | 33 | 55 | 48 | -0.5964 | 0.4934 |
| fixed_toric_PJ_R2 | oblique_residual | 55 | 55 | 48 | -0.5964 | 0.4934 |
| fixed_toric_PJ_R2 | oblique_residual | 110 | 55 | 48 | -0.5964 | 0.4934 |
| fixed_toric_PJ_R2 | oblique_residual | 225 | 55 | 48 | -0.5964 | 0.4934 |
| pruned_toric_PJ | axis_plus_residual | 5 | 33 | 48 | 0.8557 | 0.3836 |
| pruned_toric_PJ | axis_plus_residual | 15 | 33 | 48 | 0.8557 | 0.3836 |
| pruned_toric_PJ | axis_plus_residual | 33 | 33 | 48 | 0.8557 | 0.3836 |
| pruned_toric_PJ | axis_plus_residual | 55 | 33 | 48 | 0.8557 | 0.3836 |
| pruned_toric_PJ | axis_plus_residual | 110 | 33 | 48 | 0.8557 | 0.3836 |
| pruned_toric_PJ | axis_plus_residual | 225 | 33 | 48 | 0.8557 | 0.3836 |
| pruned_toric_PJ | full_table | 5 | 33 | 48 | 0.5224 | nan |
| pruned_toric_PJ | full_table | 15 | 33 | 48 | 0.5224 | nan |
| pruned_toric_PJ | full_table | 33 | 33 | 48 | 0.5224 | nan |
| pruned_toric_PJ | full_table | 55 | 33 | 48 | 0.5224 | nan |
| pruned_toric_PJ | full_table | 110 | 33 | 48 | 0.5224 | nan |
| pruned_toric_PJ | full_table | 225 | 33 | 48 | 0.5224 | nan |
| pruned_toric_PJ | oblique_residual | 5 | 33 | 48 | -0.6122 | 0.3836 |
| pruned_toric_PJ | oblique_residual | 15 | 33 | 48 | -0.6122 | 0.3836 |
| pruned_toric_PJ | oblique_residual | 33 | 33 | 48 | -0.6122 | 0.3836 |
| pruned_toric_PJ | oblique_residual | 55 | 33 | 48 | -0.6122 | 0.3836 |
| pruned_toric_PJ | oblique_residual | 110 | 33 | 48 | -0.6122 | 0.3836 |
| pruned_toric_PJ | oblique_residual | 225 | 33 | 48 | -0.6122 | 0.3836 |
| random_spectral_atoms_matched_radius | axis_plus_residual | 5 | 5 | 48 | 0.7825 | 0.0670 |
| random_spectral_atoms_matched_radius | axis_plus_residual | 15 | 15 | 48 | 0.8126 | 0.1802 |
| random_spectral_atoms_matched_radius | axis_plus_residual | 33 | 33 | 48 | 0.8417 | 0.3130 |
| random_spectral_atoms_matched_radius | axis_plus_residual | 55 | 55 | 48 | 0.8508 | 0.3617 |
| random_spectral_atoms_matched_radius | axis_plus_residual | 110 | 109 | 48 | 0.8866 | 0.5187 |
| random_spectral_atoms_matched_radius | axis_plus_residual | 225 | 225 | 48 | 0.9154 | 0.6494 |
| random_spectral_atoms_matched_radius | full_table | 5 | 5 | 48 | 0.1251 | nan |
| random_spectral_atoms_matched_radius | full_table | 15 | 15 | 48 | 0.2971 | nan |
| random_spectral_atoms_matched_radius | full_table | 33 | 33 | 48 | 0.3944 | nan |
| random_spectral_atoms_matched_radius | full_table | 55 | 55 | 48 | 0.4915 | nan |
| random_spectral_atoms_matched_radius | full_table | 110 | 109 | 48 | 0.5498 | nan |
| random_spectral_atoms_matched_radius | full_table | 225 | 225 | 48 | 0.7319 | nan |
| random_spectral_atoms_matched_radius | oblique_residual | 5 | 5 | 48 | -0.6982 | 0.0670 |
| random_spectral_atoms_matched_radius | oblique_residual | 15 | 15 | 48 | -0.6681 | 0.1802 |
| random_spectral_atoms_matched_radius | oblique_residual | 33 | 33 | 48 | -0.6390 | 0.3130 |
| random_spectral_atoms_matched_radius | oblique_residual | 55 | 55 | 48 | -0.6300 | 0.3617 |
| random_spectral_atoms_matched_radius | oblique_residual | 110 | 109 | 48 | -0.5941 | 0.5187 |
| random_spectral_atoms_matched_radius | oblique_residual | 225 | 225 | 48 | -0.5654 | 0.6494 |
| table_informed_toric_PJ_R0 | axis_plus_residual | 5 | 5 | 48 | 0.7937 | 0.1178 |
| table_informed_toric_PJ_R0 | axis_plus_residual | 15 | 15 | 48 | 0.8420 | 0.3235 |
| table_informed_toric_PJ_R0 | axis_plus_residual | 33 | 33 | 48 | 0.8723 | 0.4560 |
| table_informed_toric_PJ_R0 | axis_plus_residual | 55 | 55 | 48 | 0.9006 | 0.5821 |
| table_informed_toric_PJ_R0 | axis_plus_residual | 110 | 109 | 48 | 0.9390 | 0.7484 |
| table_informed_toric_PJ_R0 | axis_plus_residual | 225 | 225 | 48 | 0.9825 | 0.9284 |
| table_informed_toric_PJ_R0 | full_table | 5 | 5 | 48 | 0.4256 | nan |
| table_informed_toric_PJ_R0 | full_table | 15 | 15 | 48 | 0.6516 | nan |
| table_informed_toric_PJ_R0 | full_table | 33 | 33 | 48 | 0.7610 | nan |
| table_informed_toric_PJ_R0 | full_table | 55 | 55 | 48 | 0.8371 | nan |
| table_informed_toric_PJ_R0 | full_table | 110 | 109 | 48 | 0.9045 | nan |
| table_informed_toric_PJ_R0 | full_table | 225 | 225 | 48 | 0.9677 | nan |
| table_informed_toric_PJ_R0 | oblique_residual | 5 | 5 | 48 | -0.6871 | 0.1178 |
| table_informed_toric_PJ_R0 | oblique_residual | 15 | 15 | 48 | -0.6387 | 0.3235 |
| table_informed_toric_PJ_R0 | oblique_residual | 33 | 33 | 48 | -0.6084 | 0.4560 |
| table_informed_toric_PJ_R0 | oblique_residual | 55 | 55 | 48 | -0.5801 | 0.5821 |
| table_informed_toric_PJ_R0 | oblique_residual | 110 | 109 | 48 | -0.5417 | 0.7484 |
| table_informed_toric_PJ_R0 | oblique_residual | 225 | 225 | 48 | -0.4982 | 0.9284 |
| table_informed_toric_PJ_R1 | axis_plus_residual | 5 | 11 | 48 | 0.8050 | 0.1695 |
| table_informed_toric_PJ_R1 | axis_plus_residual | 15 | 11 | 48 | 0.8050 | 0.1695 |
| table_informed_toric_PJ_R1 | axis_plus_residual | 33 | 31 | 48 | 0.8314 | 0.2861 |
| table_informed_toric_PJ_R1 | axis_plus_residual | 55 | 51 | 48 | 0.8508 | 0.3728 |
| table_informed_toric_PJ_R1 | axis_plus_residual | 110 | 101 | 48 | 0.8767 | 0.4891 |
| table_informed_toric_PJ_R1 | axis_plus_residual | 225 | 221 | 48 | 0.9209 | 0.6879 |
| table_informed_toric_PJ_R1 | full_table | 5 | 11 | 48 | 0.5028 | nan |
| table_informed_toric_PJ_R1 | full_table | 15 | 11 | 48 | 0.5028 | nan |
| table_informed_toric_PJ_R1 | full_table | 33 | 31 | 48 | 0.6338 | nan |
| table_informed_toric_PJ_R1 | full_table | 55 | 51 | 48 | 0.7108 | nan |
| table_informed_toric_PJ_R1 | full_table | 110 | 101 | 48 | 0.7831 | nan |
| table_informed_toric_PJ_R1 | full_table | 225 | 221 | 48 | 0.8769 | nan |
| table_informed_toric_PJ_R1 | oblique_residual | 5 | 11 | 48 | -0.6836 | 0.1695 |
| table_informed_toric_PJ_R1 | oblique_residual | 15 | 11 | 48 | -0.6836 | 0.1695 |
| table_informed_toric_PJ_R1 | oblique_residual | 33 | 31 | 48 | -0.6634 | 0.2861 |
| table_informed_toric_PJ_R1 | oblique_residual | 55 | 51 | 48 | -0.6327 | 0.3728 |
| table_informed_toric_PJ_R1 | oblique_residual | 110 | 101 | 48 | -0.6063 | 0.4891 |
| table_informed_toric_PJ_R1 | oblique_residual | 225 | 221 | 48 | -0.5636 | 0.6879 |
| table_informed_toric_PJ_R2 | axis_plus_residual | 5 | 19 | 48 | 0.8206 | 0.2490 |
| table_informed_toric_PJ_R2 | axis_plus_residual | 15 | 19 | 48 | 0.8206 | 0.2490 |
| table_informed_toric_PJ_R2 | axis_plus_residual | 33 | 19 | 48 | 0.8206 | 0.2490 |
| table_informed_toric_PJ_R2 | axis_plus_residual | 55 | 55 | 48 | 0.8589 | 0.4168 |
| table_informed_toric_PJ_R2 | axis_plus_residual | 110 | 109 | 48 | 0.8798 | 0.5079 |
| table_informed_toric_PJ_R2 | axis_plus_residual | 225 | 217 | 48 | 0.9163 | 0.6671 |
| table_informed_toric_PJ_R2 | full_table | 5 | 19 | 48 | 0.5800 | nan |
| table_informed_toric_PJ_R2 | full_table | 15 | 19 | 48 | 0.5800 | nan |
| table_informed_toric_PJ_R2 | full_table | 33 | 19 | 48 | 0.5800 | nan |
| table_informed_toric_PJ_R2 | full_table | 55 | 55 | 48 | 0.7123 | nan |
| table_informed_toric_PJ_R2 | full_table | 110 | 109 | 48 | 0.7837 | nan |
| table_informed_toric_PJ_R2 | full_table | 225 | 217 | 48 | 0.8636 | nan |
| table_informed_toric_PJ_R2 | oblique_residual | 5 | 19 | 48 | -0.6542 | 0.2490 |
| table_informed_toric_PJ_R2 | oblique_residual | 15 | 19 | 48 | -0.6542 | 0.2490 |
| table_informed_toric_PJ_R2 | oblique_residual | 33 | 19 | 48 | -0.6542 | 0.2490 |
| table_informed_toric_PJ_R2 | oblique_residual | 55 | 55 | 48 | -0.6234 | 0.4168 |
| table_informed_toric_PJ_R2 | oblique_residual | 110 | 109 | 48 | -0.6046 | 0.5079 |
| table_informed_toric_PJ_R2 | oblique_residual | 225 | 217 | 48 | -0.5662 | 0.6671 |
| topk_dct | axis_plus_residual | 5 | 6 | 48 | 0.8655 | 0.4317 |
| topk_dct | axis_plus_residual | 15 | 16 | 48 | 0.9177 | 0.6635 |
| topk_dct | axis_plus_residual | 33 | 34 | 48 | 0.9588 | 0.8348 |
| topk_dct | axis_plus_residual | 55 | 56 | 48 | 0.9808 | 0.9238 |
| topk_dct | axis_plus_residual | 110 | 111 | 48 | 0.9969 | 0.9881 |
| topk_dct | axis_plus_residual | 225 | 226 | 48 | 1.0000 | 1.0000 |
| topk_dct | full_table | 5 | 6 | 48 | 0.6955 | nan |
| topk_dct | full_table | 15 | 16 | 48 | 0.8472 | nan |
| topk_dct | full_table | 33 | 34 | 48 | 0.9257 | nan |
| topk_dct | full_table | 55 | 56 | 48 | 0.9646 | nan |
| topk_dct | full_table | 110 | 111 | 48 | 0.9933 | nan |
| topk_dct | full_table | 225 | 226 | 48 | 1.0000 | nan |
| topk_dct | oblique_residual | 5 | 6 | 48 | -0.6153 | 0.4317 |
| topk_dct | oblique_residual | 15 | 16 | 48 | -0.5630 | 0.6635 |
| topk_dct | oblique_residual | 33 | 34 | 48 | -0.5219 | 0.8348 |
| topk_dct | oblique_residual | 55 | 56 | 48 | -0.4999 | 0.9238 |
| topk_dct | oblique_residual | 110 | 111 | 48 | -0.4838 | 0.9881 |
| topk_dct | oblique_residual | 225 | 226 | 48 | -0.4807 | 1.0000 |
| topk_dft | axis_plus_residual | 5 | 5 | 48 | 0.7937 | 0.1178 |
| topk_dft | axis_plus_residual | 15 | 15 | 48 | 0.8420 | 0.3235 |
| topk_dft | axis_plus_residual | 33 | 33 | 48 | 0.8723 | 0.4560 |
| topk_dft | axis_plus_residual | 55 | 55 | 48 | 0.9006 | 0.5821 |
| topk_dft | axis_plus_residual | 110 | 109 | 48 | 0.9390 | 0.7484 |
| topk_dft | axis_plus_residual | 225 | 225 | 48 | 0.9825 | 0.9284 |
| topk_dft | full_table | 5 | 5 | 48 | 0.4256 | nan |
| topk_dft | full_table | 15 | 15 | 48 | 0.6516 | nan |
| topk_dft | full_table | 33 | 33 | 48 | 0.7610 | nan |
| topk_dft | full_table | 55 | 55 | 48 | 0.8371 | nan |
| topk_dft | full_table | 110 | 109 | 48 | 0.9045 | nan |
| topk_dft | full_table | 225 | 225 | 48 | 0.9677 | nan |
| topk_dft | oblique_residual | 5 | 5 | 48 | -0.6871 | 0.1178 |
| topk_dft | oblique_residual | 15 | 15 | 48 | -0.6387 | 0.3235 |
| topk_dft | oblique_residual | 33 | 33 | 48 | -0.6084 | 0.4560 |
| topk_dft | oblique_residual | 55 | 55 | 48 | -0.5801 | 0.5821 |
| topk_dft | oblique_residual | 110 | 109 | 48 | -0.5417 | 0.7484 |
| topk_dft | oblique_residual | 225 | 225 | 48 | -0.4982 | 0.9284 |
| windowed_topk_dft | axis_plus_residual | 5 | 5 | 48 | 0.7825 | 0.0584 |
| windowed_topk_dft | axis_plus_residual | 15 | 15 | 48 | 0.8091 | 0.1557 |
| windowed_topk_dft | axis_plus_residual | 33 | 33 | 48 | 0.8312 | 0.2376 |
| windowed_topk_dft | axis_plus_residual | 55 | 55 | 48 | 0.8567 | 0.3430 |
| windowed_topk_dft | axis_plus_residual | 110 | 109 | 48 | 0.8940 | 0.5006 |
| windowed_topk_dft | axis_plus_residual | 225 | 225 | 48 | 0.9490 | 0.7498 |
| windowed_topk_dft | full_table | 5 | 5 | 48 | 0.3878 | nan |
| windowed_topk_dft | full_table | 15 | 15 | 48 | 0.5283 | nan |
| windowed_topk_dft | full_table | 33 | 33 | 48 | 0.6458 | nan |
| windowed_topk_dft | full_table | 55 | 55 | 48 | 0.7355 | nan |
| windowed_topk_dft | full_table | 110 | 109 | 48 | 0.8069 | nan |
| windowed_topk_dft | full_table | 225 | 225 | 48 | 0.9019 | nan |
| windowed_topk_dft | oblique_residual | 5 | 5 | 48 | -0.6983 | 0.0584 |
| windowed_topk_dft | oblique_residual | 15 | 15 | 48 | -0.6716 | 0.1557 |
| windowed_topk_dft | oblique_residual | 33 | 33 | 48 | -0.6495 | 0.2376 |
| windowed_topk_dft | oblique_residual | 55 | 55 | 48 | -0.6241 | 0.3430 |
| windowed_topk_dft | oblique_residual | 110 | 109 | 48 | -0.5868 | 0.5006 |
| windowed_topk_dft | oblique_residual | 225 | 225 | 48 | -0.5318 | 0.7498 |

Notes:

- `axis_plus_residual` fits the oblique residual and adds the best full axial projection back.
- `random_spectral_atoms_matched_radius` controls for feature budget and radial frequency profile.
- DFT variants use cos/sin atoms; DCT variants are non-periodic boundary controls.

Artifacts:

- `projection_results.csv`
- `projection_aggregate.csv`
- `residual_projection_results.csv`
- `random_matched_radius_controls.csv`
