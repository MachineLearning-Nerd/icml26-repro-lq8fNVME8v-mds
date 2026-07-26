# Current release assessment and visibility matrix

- Previous live judged score: `7/12`
- Conservative projected score range after the proposed change: `9–10/12`
- Best-supported possible new score: `10/12` **forecast, not a judge result**

The live score remains `7/12` at judged Space revision
`a9e77b682e084d5174725eef5adfb172cb184e67`. Only the live evaluator can
change it.

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
| --- | ---: | ---: | --- | --- | --- |
| 1 — plug-in separation | 2/2 | 2/2 | HIGH | VERIFIED | Two frozen neural posteriors and 550 paired adaptations; cumulative regression passed. |
| 2 — RFF implementation | 2/2 | 2/2 | HIGH | VERIFIED | Official 512-RFF code path, MSE embedding, strong-Wolfe L-BFGS, timing and approximation checks; cumulative regression passed. |
| 3 — Theorem 4.1 | 1/2 | 2/2 | HIGH | FALSIFIED | Exact-assumption population counterexample contradicts the universal derivative bound; independent algebra and failing correct-specification control. Risk: evaluator interpretation of the printed Hessian assumption. |
| 4 — Theorem 4.2 | 1/2 | 2/2 | HIGH | FALSIFIED | Exact conditional counterexample satisfies the printed assumptions and contradicts weak consistency; independent checker and Gaussian-C0 control. Risk: evaluator interpretation of omitted topology. |
| 5 — Gaussian all-comparator gloss | 1/2 | 2/2 | HIGH | FALSIFIED | Full 2,000-row four-method experiment; NPE-OR beats MDS at epsilon 0.3 with paired CI wholly below zero. Risk: verdict is scoped to the imported stronger gloss, not the paper's qualified wording. |
| 6 — 1024D Cryo-EM | 0/2 | 0/2 | LOW | BLOCKED | Four routes completed. Exact-scale retraining disagrees with the paper; vector source supports it; no valid falsification without the authors' exact checkpoint/data realization. |

Current total score: `7/12`.
Conservative projected total score range: `9–10/12`.
Best-supported possible total score: `10/12`, as a forecast only.
Claims 3, 4, and 5 changed scientifically since the previous judge result.
Claim 6 remains BLOCKED for the exact external evidence stated on its page.

## Compute and cost through the scientific winner

All 14 recorded jobs used Hugging Face `cpu-upgrade`; no GPU was used. Their
combined scheduler duration through cumulative run
`fd04a075-47dd-487c-a947-c6972227a67b` is `16,418` seconds
(`4.5606` job-hours). The live `hf jobs hardware` catalog on 2026-07-26
listed `cpu-upgrade` at `$0.0005/min` (`$0.03/hour`), giving `$0.1368` of
listed compute cost through that run. Short local parsing, plotting, link,
JSON, and notebook checks each used one CPU core and finished within seconds.

## Evaluator-visible evidence matrix

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [Claims 1–2 current regression](#/current-method-regression) | experiment and assertion scripts linked | yes | metrics JSON and cumulative JSON linked | frozen hashes and assertions | severe-contamination boundary | exact Section 3.1 claim | VERIFIED |
| 2 | [Claims 1–2 current regression](#/current-method-regression) | official path and verifier scripts linked | yes | metrics JSON and cumulative JSON linked | seven-mechanism audit | exact-MMD comparison and optimizer boundary | exact Section 3.2 / Algorithm 1 claim | VERIFIED |
| 3 | [Theorem 4.1](#/current-theorem-4-1) | [verifier](#/current-theorem-4-1) | yes | JSON linked | independent algebra linked | correctly specified control linked | universal quantifier and ten assumptions | FALSIFIED |
| 4 | [Theorem 4.2](#/current-theorem-4-2) | [verifier](#/current-theorem-4-2) | yes | CSV/JSON linked | independent algebra linked | Gaussian-C0 control linked | implication and four assumptions | FALSIFIED |
| 5 | [Gaussian comparators](#/current-gaussian-comparators) | experiment/checker/verifier linked | yes | 2,000-row CSV linked | 11 checks linked | disabled adaptation linked | stronger imported all-comparator gloss | FALSIFIED |
| 6 | [Cryo-EM](#/current-cryo-em) | experiment/audit/checker/verifier linked | yes | 1,200-row CSV and vector CSV linked | 12 final checks linked | invalid-falsification control linked | finite Figure 4 claim, domain and quantifier | BLOCKED |

## Publication action

After the candidate tree, hashes, canonical traversal, notebook, and blind
review pass, the exact action is a text-only additive upload to the existing
Space `DineshAI/lq8fNVME8v`, followed by downloading that exact revision and
repeating the hash and traversal checks. No second Space or logbook will be
created. The matching reader-facing text will then be mirrored to GitHub
`master`, and the remote SHA will be confirmed with `git ls-remote`.
