# Paper Figure Plan for V8

## Figure 1: TinyImageNet 10k Score Decomposition

Use `main_tinyimagenet_10k_scores.png` and `v8_tinyimagenet_10k_main_table.csv`.

Message: compact DCT remains the strongest full-score baseline, Toric/PJ has visible-region competence but loses full score at the boundary, hard mixed r4 repairs part of the leakage, and soft w=1 gives the best mixed full score while weakening far-band geometry.

## Figure 2: Early-Pool Causal Mechanism

Use `early_pool_causality.png` and `results/v8_tinyimagenet_mechanism_summary/tinyimagenet_mechanism_summary.csv`.

Message: early local components are sufficient and necessary. Keep L0-L3 matches or exceeds full; ablate L0-L3 collapses near zero. This should be framed as causal support for learned local chart geometry rather than merely correlational diagnostics.

## Figure 3: Local Chart Concentration vs Far-Shell Leakage

Use `cross_dataset_energy_leakage.png` and `v8_cross_dataset_theory_table.csv`.

Message: cross-dataset success tracks energy concentration in r<=4 and reduced 6<=r<9 leakage. Toric/PJ failures sit in the far-shell-heavy regime; mixed or train-radial variants move back toward local-chart concentration.

## Figure 4: Boundary-Policy Search Map

Use `boundary_policy_full_vs_far.png` and `v8_boundary_policy_ablation_table.csv`.

Message: V8 ruled out several tempting fixes. Soft w=1 is the only 10k full-score improvement over hard r4, but it does not win the far-band control. Learned chart masks are currently negative; future work should change parameterization/objective rather than continue mask-weight tuning.

## Appendix Tables

1. F1-F11 boundary-policy ablation table: `v8_boundary_policy_ablation_table.csv`.
2. C6 intervention table: `results/v8_tinyimagenet_mechanism_summary/tinyimagenet_mechanism_summary.csv`.
3. E1/E2 cross-dataset diagnostics: `v8_cross_dataset_theory_table.csv`.
4. TinyImageNet 10k anchor comparison: `v8_tinyimagenet_10k_main_table.csv`.

## Suggested Caption Claim

The experiments separate three effects that are otherwise easy to conflate: visible-region fit, far-shell extrapolation, and task-level full score. Toric/PJ learns a causally useful local geometric pool, but unconstrained far shells can dominate full extrapolation. Hard radial training gives cleaner geometry; soft radial training gives a small task-level gain; learned mask charts have not yet produced a far-shell win.
