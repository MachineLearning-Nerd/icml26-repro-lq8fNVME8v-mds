# Claim 6 LOW-confidence route record

All formal experiment nodes inherit the exact command:

```text
git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py
```

All nontrivial runs use Hugging Face `cpu-upgrade`; no GPU is used.

## Route 1 — authors' continuous posterior sampling

- Interpretation: exact full-scale released HSP90 task and continuous
  spline-flow posterior sampling used by the authors' benchmark.
- Formal run: `1df5f0be-40f8-495f-bf0c-ea9ada0ebc9f`, commit
  `fbe0df8380d3ff461c3694b28e53dd6d93d5f6da`.
- Scale: 15,000 training datasets; 10,000 RFF-fit datasets; 100 tests; 100
  32×32 images/test; 2,000 posterior samples; 1,024 RFFs; six epsilons.
- Result: clean RMSE ratio `1.436`; RMSE reductions at epsilon 0.2–0.5
  `2.8%`, `8.5%`, `8.2%`, `6.0%`. Integrity checker passed; exact claim
  verifier returned BLOCKED.
- Control: disabled adaptation gives zero gain and exits nonzero.
- Unresolved: one full-scale retraining does not verify or falsify a finite
  published experiment, and exact predictive MMD was CPU-infeasible.

## Route 2 — exhaustive discrete-state posterior

- Interpretation: the paper's parameter/prior domain is exactly the 20 HSP90
  indices, so normalize the learned density across all 20 admissible states.
- Formal run: `aebd0b74-2974-40d4-95f1-18db9d6534c6`, commit
  `cafadfb1c866f05547286414843ccb4286907e2c`.
- Result: clean RMSE ratio `0.977`; RMSE reductions at epsilon 0.2–0.5
  `21.2%`, `17.9%`, `7.6%`, `-24.6%`. All 14 integrity checks passed;
  paired/substantial-effect gates failed and the verifier returned BLOCKED.
- Control: disabled adaptation exits nonzero.
- Unresolved: this is exhaustive over the discrete task domain but differs
  from the authors' continuous-sampling evaluator.

## Route 3 — exact vector Figure 4 reconstruction

- Interpretation: reconstruct all paper-displayed means and error intervals
  from vector paths, independently calibrated by every visible y-axis tick.
- Formal run: `05d23079-27e1-4ce2-acf4-df50c9a1d351`, commit
  `4e1e16c05ece7eb7495704367cfe30792734b83e`.
- Result: all 48 rows and 12 integrity checks passed. Paper-displayed RMSE
  reductions are `63.0%`, `58.8%`, `58.9%`, `58.4%`; predictive-MMD
  reductions are `9.6%`, `11.9%`, `13.1%`, `12.9%`; clean ratios pass.
- Control: assigning NPE to disabled MDS gives zero gain and exits nonzero.
- Unresolved: this verifies what the source figure displays, not the
  independent underlying experiment, so the verifier returns BLOCKED.

## Route 4 — mandatory falsification qualification

- Interpretation: a valid counterexample must satisfy the exact task,
  algorithm, contamination, scale, and estimand and contradict the exact
  source quantifier.
- Evidence: Route 1's 1,200 logged rows reconstruct the original CSV SHA-256
  exactly: `ef27c8f5d79c1cc225d1fba328c3f4de86f0b3bf914b7da1de47797dac3150de`.
- Candidates: full-scale independent retraining; worst valid single query;
  disabled-MDS/altered-noise assumption violation.
- Qualification result: none contradicts a universal source quantifier.
  The paper reports a finite Figure 4 experiment and does not quantify over
  every retraining seed, realization, test set, or query.
- Control: deliberately treating a failed reproduction as falsification must
  exit nonzero.
- Formal run: `6e01e046-c72a-4ddc-8a83-daf8d95be54d`, commit
  `1d7e7cad0bd13410d48cfa5dfe69604cf06e4a5f`.
- Result: all 12 independent checks passed, the reconstructed Route 1 CSV
  matched SHA-256
  `ef27c8f5d79c1cc225d1fba328c3f4de86f0b3bf914b7da1de47797dac3150de`,
  no candidate qualified as a valid falsification, and the verifier exited
  nonzero with the final verdict `BLOCKED`.
- Unblocker: the authors' exact checkpoint and training/test realization, or
  an explicit universal quantifier over seeds/realizations.
