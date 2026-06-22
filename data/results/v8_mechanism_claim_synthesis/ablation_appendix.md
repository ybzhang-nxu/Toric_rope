# V8 Ablation Appendix Summary

## Boundary Policy

Hard train-radial r4 remains the cleanest geometry-preserving boundary policy. On TinyImageNet 10k it reaches final 0.8449, visible 0.8658, and far band 0.2565.

Soft radial w=1 is the only V8 policy that improves TinyImageNet 10k full score over hard r4: final rises to 0.8507 and the full-visible gap improves to -0.0137. Its far band falls to 0.2339, so this is a task-level improvement rather than a far-shell geometry success.

## Negative Controls

- r5 train radius collapses full extrapolation even when visible-region scores look healthy.
- Scalar radial-tail penalties do not solve boundary leakage and worsen the full-visible gap.
- LC/log direct coordinate warps can fit the visible region but fail full extrapolation.
- top16 structural compression is destructive.
- Learned chart masks learn weights, and smooth+monotonic regularization makes those weights well-behaved, but the task and far-band gates still fail.

## Mechanism Interpretation

The robust mechanism is not simply “more capacity” or “more visible offsets.” It is the interaction between a local chart, early-layer storage of local geometry, and a boundary policy that prevents far-shell leakage from dominating full extrapolation.
