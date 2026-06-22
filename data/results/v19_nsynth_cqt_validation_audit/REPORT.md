# MT-A07 NSynth/CQT Validation Audit

This report re-reads the NSynth/CQT result CSV files and recomputes the key
claims used by the real-data validation narrative.  It is an audit layer,
not a new training run.

## Summary

- wav files: `256`
- checks: `20`
- pass: `17`
- boundary: `3`
- fail: `0`
- review: `0`

## Checks

| gate | status | value | criterion |
|---|---|---|---|
| data_count | pass | 256 | at least 256 real NSynth wav files are present |
| v14_downstream_boundary | boundary | best=axis_additive:-0.0258; J2=-0.0421 | MT-A07 should not be described as a downstream J2 win |
| v15_empirical_J1_minus_J0 | pass | 0.0706 | J1 improves over J0 on empirical CQT offset table |
| v15_empirical_J2_minus_J1 | pass | 0.0603 | J2 improves over J1 on empirical CQT offset table |
| v15_empirical_J2_minus_shuffle | pass | 0.8529 | coordinate shuffle collapses empirical CQT Toric/PJ fit |
| v16_subset128_J1_minus_J0 | pass | min=0.0697, mean=0.0735 | J1-J0 is positive across random 128-example real-data subsets |
| v16_subset128_J2_minus_J1 | pass | min=0.0541, mean=0.0612 | J2-J1 is positive across random 128-example real-data subsets |
| v16_subset128_J2_minus_shuffle | pass | min=0.8559, mean=0.9333 | J2-shuffle is positive across random 128-example real-data subsets |
| v16_subset192_J1_minus_J0 | pass | min=0.0695, mean=0.0735 | J1-J0 is positive across random 192-example real-data subsets |
| v16_subset192_J2_minus_J1 | pass | min=0.0587, mean=0.0624 | J2-J1 is positive across random 192-example real-data subsets |
| v16_subset192_J2_minus_shuffle | pass | min=0.8483, mean=0.9158 | J2-shuffle is positive across random 192-example real-data subsets |
| v17_learned_mean_hierarchy | pass | J0=0.8223, J1=0.8913, J2=0.9297, shuffle=0.0316 | learned mean tables satisfy J0 < J1 < J2 and shuffle collapse |
| v17_alltable_J1_minus_J0 | pass | mean=0.0820, min=0.0596, positive=15/15 | learned head+mean tables have positive J1-J0 margins |
| v17_alltable_J2_minus_J1 | pass | mean=0.0439, min=0.0254, positive=15/15 | learned head+mean tables have positive J2-J1 margins |
| v17_alltable_J2_minus_shuffle | pass | mean=0.8935, min=0.8447, positive=15/15 | learned head+mean tables have positive J2-shuffle margins |
| v18_random_holdout_J1_minus_J0 | pass | heldout mean=0.0734, min=0.0240, positive=45/45 | random heldout offsets preserve positive J1-J0 margin at ridge 1e-6 |
| v18_random_holdout_J2_minus_J1 | pass | heldout mean=0.0442, min=0.0123, positive=45/45 | random heldout offsets preserve positive J2-J1 margin at ridge 1e-6 |
| v18_random_holdout_J2_minus_shuffle | pass | heldout mean=0.9590, min=0.8686, positive=45/45 | random heldout offsets preserve positive J2-shuffle margin at ridge 1e-6 |
| v18_outer_shell_regularized_boundary | boundary | J2-J1 mean=0.7528, min=0.3471; J2-shuffle mean=1.1378, min=0.6606 | outer-shell deployment needs explicit ridge/boundary handling |
| v18_checkerboard_aliasing_stress | boundary | J2-shuffle mean=-0.6200, positive=2/15 | checkerboard split is not a clean coordinate-sensitive positive result |

## Reading

The audited evidence supports a bounded claim: real NSynth/CQT scalar
offset fields and learned scalar attention-bias tables exhibit a stable
coordinate-sensitive J0 -> J1 -> J2 hierarchy under projection and
random heldout-offset checks.  The downstream masked-reconstruction
pilot remains a boundary result, and structured heldout splits reinforce
the need for explicit deployment-tail checks.

Artifacts:

- `evidence_checks.csv`
- `audited_nsynth_margins.pdf`
- `summary.json`
