# Branch audit and lineage

The old branches were created during an OpenResearch campaign and used the
temporary orx/ prefix. They are renamed by role so that a reader can infer
what a branch contains without knowing the experiment tooling. No experiment
result is changed by the rename.

All branch commits are normalized to:

MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>

The hashes below are the canonicalized source tips before the documentation
commit on main. The final branch tip may differ only where a documentation
change was intentionally propagated.

| Canonical public branch | Legacy branch | Purpose | Producer or evidence | Result |
| --- | --- | --- | --- | --- |
| main | master | Reader-facing landing page, ledger, report, and final docs | README, claims.json, report, committed outputs | Documentation surface; not an experiment |
| baseline/judged-7-of-12 | orx/frozen-judged-baseline-with-uv-lock | Freeze the judged baseline and CPU environment | pyproject.toml, uv.lock, reproduction/ENVIRONMENT.md | Baseline preserved; last live score 7/12 |
| baseline/portable-cumulative | orx/portable-cumulative-baseline | Make cumulative timing gates portable across CPU hosts | reproduction/test_reproduction.py and run_all.py | Portable release gate |
| audit/claim-3-theorem-4-1 | orx/falsify-theorem-4-1-exact-assumptions | Test Theorem 4.1 under all ten printed assumptions | theorem_4_1_counterexample.py and formal checker | Claim 3 FALSIFIED_AS_WRITTEN |
| audit/claim-4-theorem-4-2 | orx/falsify-theorem-4-2-exact-assumptions | Test Theorem 4.2 under its printed assumptions | theorem_4_2_counterexample.py and formal checker | Claim 4 FALSIFIED_AS_WRITTEN |
| proof/claims-3-4-integrated | orx/integrated-exact-theorem-falsifications | Integrate both formal theorem counterexamples | reproduction/campaign and .openresearch artifacts | Claims 3 and 4 preserved in cumulative evidence |
| experiment/gaussian-full-comparators | orx/full-gaussian-comparator-reproduction | Add NPE, NNPE, NPE-OR, and MDS comparison | gaussian_full_comparators.py and claim-5 artifact | Claim 5 stronger gloss FALSIFIED |
| experiment/cryo-em-full-1024d | orx/full-1024d-cryo-em-reproduction | Retrain the 1024-dimensional Cryo-EM route at reported scale | cryo_full_reproduction.py and claim-6 artifact | Primary route complete; final claim BLOCKED |
| experiment/cryo-discrete-posterior | orx/cryo-discrete-state-posterior-verification | Exhaustively evaluate all 20 Cryo posterior states | run_cryo_claim.py and discrete-state route | Gates do not resolve the discrepancy; BLOCKED |
| audit/cryo-figure-4-reconstruction | orx/cryo-figure-4-vector-reconstruction | Reconstruct all displayed vector Figure 4 values | cryo_figure4_reconstruction.py | 48/48 values and 12/12 checks; source-only evidence |
| audit/cryo-falsification-qualification | orx/cryo-mandatory-falsification-qualification | Reject invalid alternatives as universal counterexamples | cryo_falsification_audit.py | No valid falsification; Claim 6 remains BLOCKED |
| release/integrated-claims-1-6 | orx/integrated-claims-1-6-cumulative-evidence | Scientific winner integrating all six claim paths | run_all.py and cumulative artifacts | Claims 1–2 VERIFIED, 3–5 FALSIFIED, 6 BLOCKED |
| release/evaluator-visible | orx/evaluator-visible-existing-logbook-release | Preserve the evaluator-facing existing logbook release | .trackio/logbook and validate_release.py | Reader-facing release surface |

## Canonicalized source tips

| Canonical branch | Source tip after identity normalization |
| --- | --- |
| baseline/judged-7-of-12 | 507bd9f84d8f3b3ec0fda79ef74905821c147e35 |
| baseline/portable-cumulative | 5562fe3b02a02273d3555846e335334fb2a49f54 |
| audit/claim-3-theorem-4-1 | 1bc25570ef33769a14fb80ff4f81ee2ac7092310 |
| audit/claim-4-theorem-4-2 | ce9229db45cd09564f09c3203d362887a813911b |
| proof/claims-3-4-integrated | 31b02f4ae8ced6d696c704a2b5ac096f9c787b82 |
| experiment/gaussian-full-comparators | 495d9afd6fc9aefc1e73fa12eb3bbde678cd4a6c |
| experiment/cryo-em-full-1024d | 4340cc3283844b22b296947f8b19ae3d26c958a8 |
| experiment/cryo-discrete-posterior | 18c3d7b38c8f6b07b005d6a251126eb0d0bc910f |
| audit/cryo-figure-4-reconstruction | 322e411eabac6c1d731171f69ce3f8cc7d9004bd |
| audit/cryo-falsification-qualification | 4eb104a2acefbfcf9a0393c0c2218d64ecc0b807 |
| release/integrated-claims-1-6 | 9d35e6c0dd538e7a9eefd413e8ee6ab128333812 |
| release/evaluator-visible | ed4d36f96cebe2e9748de08e71d2f7c92d4f99b1 |

## Naming policy

- main is the only default branch.
- baseline branches freeze environment or regression baselines.
- experiment branches produce new empirical evidence.
- proof branches contain formal proof or theorem checks.
- audit branches qualify, reconstruct, or independently challenge evidence.
- release branches are assembled, evaluator-facing publication surfaces.

The final public branch set contains exactly the 13 canonical names above.
