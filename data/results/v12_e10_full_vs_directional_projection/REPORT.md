# V12 E10 Full Multivariate Versus Directional Jets

Date: 2026-06-21

Source projection artifact:

```text
MetricToric/results/v12_e1_teacher_jetorder_projection_cifar10_10k/projection_aggregate.csv
```

This audit extracts the full-vs-directional comparison already present in the E1 teacher jet-order projection run. No model is retrained.

## Main Matched-Atom Result

Across matched-atom J1/J2 rows, full multivariate jets improve target-fit R2 over directional jets in 24/24 comparisons. The mean delta is 0.1101.

Full-table target, atom budget 108:

| source | order | dir R2 | full R2 | delta | dir cond | full cond | cond ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generic | J1 | 0.7873 | 0.8675 | 0.0802 | 2.84e+12 | 6.12e+05 | 2.15e-07 |
| generic | J2 | 0.8077 | 0.8629 | 0.0552 | 2.46e+12 | 2.04e+06 | 8.31e-07 |
| table_informed | J1 | 0.7697 | 0.8453 | 0.0756 | 3.20e+12 | 2.59e+12 | 8.10e-01 |
| table_informed | J2 | 0.7714 | 0.8155 | 0.0440 | 3.47e+12 | 3.10e+12 | 8.95e-01 |

Axis-plus-residual target, atom budget 108:

| source | order | dir R2 | full R2 | delta | dir cond | full cond | cond ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generic | J1 | 0.5407 | 0.6345 | 0.0938 | 2.84e+12 | 6.12e+05 | 2.15e-07 |
| generic | J2 | 0.5594 | 0.6258 | 0.0664 | 2.46e+12 | 2.04e+06 | 8.31e-07 |
| table_informed | J1 | 0.4589 | 0.6052 | 0.1464 | 2.83e+12 | 2.31e+12 | 8.18e-01 |
| table_informed | J2 | 0.4829 | 0.5440 | 0.0611 | 3.16e+12 | 2.87e+12 | 9.07e-01 |

Full-table target, atom budget 54:

| source | order | dir R2 | full R2 | delta | dir cond | full cond | cond ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generic | J1 | 0.6595 | 0.7943 | 0.1348 | 1.87e+12 | 2.61e+02 | 1.39e-10 |
| generic | J2 | 0.6631 | 0.7946 | 0.1314 | 2.68e+12 | 1.00e+04 | 3.74e-09 |
| table_informed | J1 | 0.6949 | 0.7466 | 0.0518 | 2.70e+12 | 2.29e+12 | 8.48e-01 |
| table_informed | J2 | 0.5711 | 0.7008 | 0.1297 | 2.92e+12 | 2.63e+12 | 9.00e-01 |

Axis-plus-residual target, atom budget 54:

| source | order | dir R2 | full R2 | delta | dir cond | full cond | cond ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generic | J1 | 0.2896 | 0.4976 | 0.2081 | 1.87e+12 | 2.61e+02 | 1.39e-10 |
| generic | J2 | 0.2320 | 0.4035 | 0.1715 | 2.68e+12 | 1.00e+04 | 3.74e-09 |
| table_informed | J1 | 0.3514 | 0.4115 | 0.0601 | 2.40e+12 | 2.03e+12 | 8.49e-01 |
| table_informed | J2 | 0.2327 | 0.3951 | 0.1624 | 2.66e+12 | 2.44e+12 | 9.19e-01 |

## Nested Same-Center Note

The nested rows compare the same center schedule but not the same atom budget: directional nested rows use 109 nominal atoms and full nested rows use 73. They are useful for conditioning diagnostics, but the matched-atom table above is the fairer E10 comparison.

In nested same-center rows, full multivariate jets improve target-fit R2 in 0/12 comparisons and tie within 0.0001 in 12/12 comparisons; the mean delta is -0.0000.

## Downstream Boundary

The projection audit should be read together with the matched downstream confirmation. Full multivariate J1/J2 project better than directional J1/J2, but the natural CIFAR10 downstream task still prefers the matched J0 row:

| basis | n | score mean | score std | final mean | teacher-init R2 note |
| --- | ---: | ---: | ---: | ---: | --- |
| full multivariate J0 | 3 | 0.8634 | 0.0011 | 0.8636 | 0.8397 from E1 confirm3 |
| full multivariate J1 | 3 | 0.8610 | 0.0016 | 0.8609 | 0.8155 from J1/J2 confirm3 |
| full multivariate J2 | 3 | 0.8575 | 0.0006 | 0.8578 | 0.7991 from J1/J2 confirm3 |

## Interpretation

- Full multivariate jets are a better projection family than directional jets for the learned teacher table under matched atom budgets.
- This supports retaining the full multivariate PJ construction as a theory/diagnostic object.
- It does not create a monotonic downstream jet-order claim: the matched downstream confirmation still shows J0 > J1 > J2 for CIFAR10 reconstruction.
- The paper should separate projection expressivity from downstream utility.

## Artifacts

```text
matched_atom_full_vs_directional.csv
nested_same_center_full_vs_directional.csv
REPORT.md
```
