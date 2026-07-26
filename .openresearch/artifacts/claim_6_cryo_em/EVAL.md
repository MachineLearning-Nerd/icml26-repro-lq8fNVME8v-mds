# Claim 6 evaluation

Status: ROUTE 2 PENDING; ROUTE 1 BLOCKED

Route 1 completed at exact paper scale on Hugging Face `cpu-upgrade`: 15,000
training datasets, 10,000 RFF-fit datasets, 100 tests, 100 32×32 images per
test, 2,000 posterior samples, and all six contamination levels. The
independent integrity checker passed all checks and the NPE tensor hash was
unchanged.

The authors' continuous-sampling interpretation did not satisfy the
preregistered contract. Mean RMSE for NPE versus MDS was respectively
`4.073/5.849` at epsilon 0 and `6.004/5.837`, `6.395/5.854`,
`6.380/5.860`, and `6.278/5.899` at epsilon 0.2–0.5. The clean ratio was
`1.436`; mean robustness reductions were only `2.8%–8.5%`, below the
preregistered 30% threshold. The verdict is honestly BLOCKED, not falsified.

Route 2 retains every continuous result and additionally normalizes the
learned density over the complete discrete state domain `{0,...,19}`. The
current verifier targets those exhaustive grid metrics and remains fail-closed.
The disabled-adaptation control must still exit nonzero. No forecast point is
assigned before the formal Route 2 run.
