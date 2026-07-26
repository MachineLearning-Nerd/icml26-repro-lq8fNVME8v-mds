# Current verification — 1024-dimensional Cryo-EM claim

**Verdict: BLOCKED. Confidence: LOW.** Four materially different routes were
completed. The paper-displayed result is clear, but the independent exact-scale
retraining did not reproduce its magnitude, and the authors' exact Figure 4
checkpoint and data realization are unavailable. The discrepancy is not
mislabelled as falsification.

## Exact claim, domain, and quantifier

Section 6.3 states that, on the HSP90 task, “Despite the high dimensionality
and severe nature of the contamination, MDS is able to substantially improve
the robustness of the NPE to contamination (Figure 4).”

Domain: 20-state discrete-uniform HSP90 prior; datasets of 100 32×32 images;
15,000 training datasets; 100 test datasets; epsilon
`{0,.1,.2,.3,.4,.5}`; normalized pure-Gaussian-noise image replacement.
Source: arXiv 2602.09161, TeX lines 423–433, Section 6.3 and Figure 4. TeX
SHA-256:
`ff81fd973e3bcba86fb23e9a0c102ec88e240f62361315c7875de54e29ea4fd2`.

This is a finite empirical report tied to Figure 4. It does not universally
quantify over every training seed, simulator realization, valid test set, or
individual query.

## Four verification routes

| Route | Defensible interpretation | Result |
| --- | --- | --- |
| 1 | Authors' continuous spline-flow posterior sampling at exact scale | Clean MDS/NPE RMSE ratio `1.436`; only `2.8–8.5%` reductions at epsilon 0.2–0.5. Integrity passed; claim gate BLOCKED. |
| 2 | Normalize learned density exhaustively over all 20 admissible states | Clean ratio `0.977`; reductions `21.2%, 17.9%, 7.6%, -24.6%`. All 14 checks passed; effect gates failed. |
| 3 | Reconstruct the exact vector Figure 4 using independently calibrated axis ticks | All 48 values and 12 checks passed. The paper displays `58–63%` RMSE and `9.6–13.1%` predictive-MMD reductions, but this is source evidence, not an independent experiment. |
| 4 | Mandatory falsification qualification | Three candidate counterexamples audited; none both satisfies the exact contract and contradicts the finite source quantifier. Final status BLOCKED. |

Route 1 mean posterior RMSE:

| epsilon | NPE | NPE-MDS | reduction |
| ---: | ---: | ---: | ---: |
| 0.0 | 4.0727 | 5.8490 | -43.6% |
| 0.2 | 6.0041 | 5.8368 | 2.8% |
| 0.3 | 6.3951 | 5.8542 | 8.5% |
| 0.4 | 6.3797 | 5.8597 | 8.2% |
| 0.5 | 6.2781 | 5.8994 | 6.0% |

The mandatory route rejected three invalid falsification shortcuts: a
different full-scale retraining does not contradict a finite saved
realization; the worst individual query does not contradict the 100-test
aggregate; and disabled MDS or altered noise violates the named algorithm or
contamination.

## Raw evidence, executable verifier, and control

Download:
[Route 1 raw 1,200 rows](route_1_raw_trials.csv),
[Route 1 summary](route_1_summary.json),
[Route 3 vector reconstruction](figure4_digitized.csv),
[Route 4 falsification audit](falsification_audit.json),
[independent checker](independent_checker_output.json),
[verifier output](verifier_output.json),
[negative-control output](negative_control_output.json),
[compute receipt](runtime.json), and
[all four attempt records](attempts.md).

Executable source:
[full-scale experiment](cryo_full_reproduction.py),
[falsification audit](cryo_falsification_audit.py),
[independent checker](check_cryo_falsification.py), and
[fail-closed verifier](verify_cryo_falsification.py).

The Route 1 raw SHA-256 is
`ef27c8f5d79c1cc225d1fba328c3f4de86f0b3bf914b7da1de47797dac3150de`.
Route 4 reproduced all 12 RMSE aggregates and passed all 12 independent
checks. Its verifier exited `1` with `BLOCKED`. The negative control—treating
a failed reproduction as a valid falsification—also exited `1`.

## Command, compute, deviations, and unblocker

Every route inherited:

```text
git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py
```

The exact-scale Route 1 formal run was
`1df5f0be-40f8-495f-bf0c-ea9ada0ebc9f`, Git `fbe0df8380d3ff461c3694b28e53dd6d93d5f6da`,
Hugging Face `cpu-upgrade`, 64 allocated logical CPUs, no GPU, 3,011
scientific seconds. The final cumulative qualification reran in
`fd04a075-47dd-487c-a947-c6972227a67b` at Git
`41ec2e32d267e6944c04cc766c495f59a3071fdb`.

Deviation: Route 1's predictive distance uses a preregistered RFF estimator,
not the paper's CPU-infeasible exact quadratic five-bandwidth MMD; posterior
RMSE is the primary exact metric. The simulator's released seed argument is
ineffective, so deterministic corpus hashes were audited.

Unblocker: the authors' exact saved NPE checkpoint and exact Figure 4
training/test corpora and realization, or an explicit universal quantifier
over retraining seeds/realizations.
