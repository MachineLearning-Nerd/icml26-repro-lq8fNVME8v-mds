# Claim 6 evaluation

Status: BLOCKED

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

Route 3 reconstructed all 48 means and intervals in the exact vector Figure 4
using only axis-tick calibration. The independent checker passed. The paper
curves show 58–63% RMSE and 9.6–13.1% predictive-MMD reductions at epsilon
0.2–0.5 while preserving the clean regime, but the route correctly remains
BLOCKED because paper-source evidence is not an independent experiment.

Confidence remained LOW after three routes, so the mandatory fourth
falsification route was completed. It reconstructed Route 1's complete raw
CSV from the formal log with the exact SHA-256
`ef27c8f5d79c1cc225d1fba328c3f4de86f0b3bf914b7da1de47797dac3150de`,
reproduced all 12 aggregate RMSE means, and passed all 12 independent
integrity checks. The formal Route 4 run was
`6e01e046-c72a-4ddc-8a83-daf8d95be54d`.

None of the three counterexample candidates both satisfied the complete
claim contract and contradicted the paper's exact finite empirical
quantifier. The full-scale seed-42 discrepancy is important evidence, but the
paper does not universally quantify over every retraining seed or test
realization; the worst individual query does not contradict the 100-test
aggregate; and disabled adaptation or altered noise violates the named
algorithm or contamination assumptions. The fail-closed verifier therefore
exited nonzero with `BLOCKED`, and the negative control also exited nonzero
when asked to mislabel a failed reproduction as falsification.

Unblocker: the authors' exact saved NPE checkpoint and exact training/test
corpora and realization for Figure 4, or an explicit universal quantifier
over retraining seeds/realizations that the full-scale counterexample can
legitimately contradict.
