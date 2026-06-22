# V9 Far-Band Teacher 10k Summary

Candidate: `farband_teacher_w3_10k_3seed`

| metric | value | gate | pass |
|---|---:|---:|---|
| full | 0.8676 +/- 0.0071 | > 0.8507 | True |
| r6-9 | 0.2762 +/- 0.0127 | > 0.2565 | True |
| r0-4 | 0.8727 +/- 0.0031 | >= 0.8658 | True |
| gap | -0.0053 | >= -0.0137 | True |
| energy r<=4 | 0.7675 | high/localized | True |
| energy r6-9 | 0.0297 | lower than hard-r4 0.1332 | True |

Per-seed full scores:

| seed | full | visible | gap |
|---:|---:|---:|---:|
| 426 | 0.8737 | 0.8761 | -0.0025 |
| 526 | 0.8577 | 0.8687 | -0.0109 |
| 626 | 0.8715 | 0.8741 | -0.0026 |

Decision: strong 10k success on the targeted far-band teacher objective. This closes the scalar-bias targeted repair branch, but it should still be described as teacher-signal repair rather than a pure unsupervised continuation proof.

Diagnostics: `results/v9_curvature_diagnostics_10k/`.
