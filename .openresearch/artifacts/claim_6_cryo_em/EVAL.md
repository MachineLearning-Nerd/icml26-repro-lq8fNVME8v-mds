# Claim 6 evaluation

Status: ROUTE 3 PENDING; ROUTES 1–2 BLOCKED

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

Route 2 retained every continuous result and normalized the learned density
over the complete state domain `{0,...,19}`. It preserved the clean regime
(RMSE ratio `0.977`) and improved epsilon 0.2–0.4, but the relative RMSE gains
were only `21.2%`, `17.9%`, `7.6%`, and `-24.6%` at epsilon 0.2–0.5. Its
independent checker passed; its fail-closed verifier correctly returned
BLOCKED.

Route 3 reconstructs all means and intervals in the exact vector Figure 4
using only axis-tick calibration. It is source-level corroboration, not an
independent experiment, and is preregistered to remain BLOCKED even if the
paper curves satisfy every effect threshold. No forecast point is assigned
before its formal run.
