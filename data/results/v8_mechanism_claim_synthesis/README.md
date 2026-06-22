# V8 Mechanism Claim Synthesis

This directory collects the V8 mechanism-level synthesis after F11.

## Core Claim

Toric/PJ learns useful local and oblique geometry, but TinyImageNet full-score success depends on how the model treats the boundary and far shells. The strongest causal evidence is C6: the early L0-L3 local pool is sufficient to match or exceed the full model, while ablating that pool collapses close to the zero baseline. The strongest diagnostic evidence is E1/E2: success tracks local-chart energy concentration versus far-shell leakage, not visible-region fit alone.

## Main Takeaways

1. Early local geometry is causal: Toric/PJ L0-L3 keep reaches 0.8475 versus full 0.8279, while L0-L3 ablate falls to 0.3037 near zero 0.2790. Mixed r4 shows the same pattern: keep 0.8665 versus full 0.8521, ablate 0.2698 near zero 0.2484.
2. Hard train-radial r4 is the cleaner far-shell geometry baseline: TinyImageNet hard r4 10k has far band 0.2565, higher than soft w=1 10k at 0.2339.
3. Soft radial w=1 is a real task-level success candidate: TinyImageNet 10k final improves from hard r4 0.8449 to soft w=1 0.8507, and the full-visible gap improves from -0.0209 to -0.0137.
4. Learned chart masks are negative so far: radial-only, radial-angular, quadrant-tied, and smooth+monotonic all learn nontrivial mask structure but fail the task/far-band gates.
5. Negative controls reject simple alternatives: r5 training radius collapses, scalar far-tail penalties worsen extrapolation, LC/log direct coordinate warps are visible-good but full-bad, and top16 structural compression is destructive.

## Files

- `v8_mechanism_claim_table.csv`: compact claim/evidence/verdict table.
- `v8_tinyimagenet_10k_main_table.csv`: TinyImageNet 10k main comparison with far-band controls.
- `v8_boundary_policy_ablation_table.csv`: F1-F11 boundary-policy appendix table.
- `v8_cross_dataset_theory_table.csv`: E1/E2 theory diagnostics across datasets.
- `main_tinyimagenet_10k_scores.png`: TinyImageNet 10k score decomposition.
- `boundary_policy_full_vs_far.png`: boundary-policy search map.
- `early_pool_causality.png`: C6 early-pool intervention figure.
- `cross_dataset_energy_leakage.png`: local-energy versus far-shell leakage figure.
- `paper_figure_plan.md`: paper-style main figure and appendix plan.
- `ablation_appendix.md`: prose summary of V8 ablations.
