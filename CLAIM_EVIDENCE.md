# Claim-to-evidence audit

This file explains how each claim is generated, checked, and bounded. A
verdict applies only to the claim contract written here; nearby wording is not
silently substituted.

## Common evidence pipeline

The campaign uses six layers:

1. The arXiv PDF and source TeX define the paper claim.
2. The official implementation is pinned as a git submodule.
3. A producer script generates raw observations or a formal counterexample.
4. An independent checker reconstructs the reported summaries.
5. A negative control tests a necessary boundary or alternative explanation.
6. The release records the verdict, hashes, limitations, and branch lineage.

The primary machine-readable ledger is [claims.json](claims.json). The
principal output hashes are in [EVIDENCE_MANIFEST.json](EVIDENCE_MANIFEST.json).

## Claim 1 — frozen plug-in separation

Claim contract: MDS changes the test-time query summary while the pretrained
neural posterior estimator remains fixed.

Producer path:

- reproduction/reproduce_mds.py implements the controlled Gaussian adapter.
- reproduction/neural_npe_upgrade.py evaluates frozen Gaussian and OUP neural
  posteriors.
- reproduction/integrate_neural_upgrade.py merges the neural evidence into the
  cumulative release.

Evidence path:

- outputs/full/neural_upgrade_summary.json records frozen-posterior gates.
- outputs/full/neural_gaussian_aggregate.csv records 50 paired trials at each
  contamination level for the Gaussian neural posterior.
- outputs/full/neural_oup_aggregate.csv records the second mechanism.
- The claim-1 OpenResearch artifact stores the contract, method, raw results,
  checker, and negative control.

Independent checks:

- reproduction/test_reproduction.py checks posterior files, parameter counts,
  finite aggregates, positive paired gains, and adaptation timing.
- The frozen tensor hashes are checked before and after adaptation.
- The severe-contamination rows are retained rather than hidden.

Verdict: VERIFIED_SCOPED. The evidence supports the modular frozen-NPE
mechanism on the recorded Gaussian and OUP tasks. It does not establish the
same result for every task in the paper.

## Claim 2 — RFF/MSE/L-BFGS implementation

Claim contract: the method uses Gaussian-kernel MMD approximated with random
Fourier features, an offline MSE-trained conditional embedding network, and
test-time strong-Wolfe L-BFGS.

Producer path:

- reproduction/reproduce_mds.py loads the pinned official rff.py module
  directly, avoiding the official package initializer's undeclared sbibm
  import.
- reproduction/audit_dimension_general_proof.py audits the proof source and
  dimension scope.

Evidence path:

- outputs/full/source_and_code_audit.json records code signatures, source
  hashes, and packaging caveats.
- outputs/full/experiment_summary.json records RFF dimension, regressor,
  optimizer, seeds, timing, and exact-versus-RFF comparison.
- The claim-2 artifact contains the claim contract and independent checks.

Important configuration boundary: the paper and reproduction use 512 RFFs,
while the official repository default is 256. The override is explicit in the
producer and is not presented as an untouched default.

Verdict: VERIFIED_SCOPED. The implementation path and lightweight CPU
behavior are supported for the recorded setup; this is not a claim that every
optional package import works without the disclosed dependency caveat.

## Claim 3 — Theorem 4.1

Claim contract: under the ten Appendix A.1 assumptions, the supremum over
contamination points of the right derivative of posterior KL is finite.

Producer path:

- reproduction/campaign/theorem_4_1_counterexample.py constructs the exact
  Gaussian decoder, bounded kernel, summary domain, symmetric target, and
  point-mass contamination.
- run_theorem_4_1.py runs the formal route.
- check_theorem_4_1_independent.py reconstructs constants and rates.

Evidence path:

- formal_results.json records the FALSIFIED verdict.
- formal_independent_checker_output.json reports PASS for all-ten-assumption
  audit, positive paper model-averaged Hessian, zero target Hessian, cube-root
  summary rate, and KL quotient divergence.
- formal_negative_control_output.json records the correctly specified control.

Result: the target objective has a quartically flat minimum. The quotient
KL/epsilon increases from 20.87 at epsilon 1e-3 to 4547.80 at epsilon 1e-10.

Verdict: FALSIFIED_AS_WRITTEN. The counterexample attacks the theorem's
universal statement under the printed assumptions. It does not imply that
ordinary practical objectives are quartically flat.

## Claim 4 — Theorem 4.2

Claim contract: under the four Appendix A.3 assumptions, posterior weak
consistency based on the original summary implies weak consistency after MDS.

Producer path:

- reproduction/campaign/theorem_4_2_counterexample.py constructs the exact
  conditional model on a closed, locally compact countable subset.
- run_theorem_4_2.py executes the formal route.
- check_theorem_4_2_independent.py verifies the kernel energy and escaping
  summary argument.

Evidence path:

- formal_result.json records the FALSIFIED verdict.
- formal_results.csv contains the finite-sample contraction and escaping
  summary values.
- formal_independent_checker_output.json reports PASS for all printed
  assumptions, exact objectives, original consistency, MDS inconsistency, and
  mass escape.
- formal_negative_control_output.json records the Gaussian-C0 control.

Result: the original posterior Wasserstein distance falls from 6.67e-2 at
sample size 3 to 6.21e-11 at 30, while the MDS objective selects an escaping
point mass. The missing condition is stronger topology than bounded ISPD
alone.

Verdict: FALSIFIED_AS_WRITTEN. The result is scoped to the theorem's universal
assumptions and does not reject compact or C0-controlled special cases.

## Claim 5 — Gaussian comparator scope

Claim contract: the stronger imported release gloss says that MDS beats every
named comparator below 40% contamination. This is intentionally distinguished
from the paper's qualified in-general wording.

Producer path:

- reproduction/campaign/gaussian_full_comparators.py generates the complete
  NPE, NNPE, NPE-OR, and MDS factorial.
- run_gaussian_comparators.py runs the route.
- check_gaussian_comparators.py reconstructs the aggregate and checks hashes.

Evidence path:

- outputs/full/gaussian_trials.csv contains 300 controlled MDS rows.
- outputs/full/gaussian_aggregate.csv contains the aggregate MDS metrics.
- The claim-5 artifact contains the 2,000-row four-method raw CSV, aggregate,
  checker, verifier, and disabled-adaptation control.

Result: MDS beats ordinary NPE and NNPE at 10–30% contamination. At 30%,
NPE-OR has MMD 1.0626 versus MDS 2.0756; the paired 95% interval for
NPE-OR minus MDS is [-1.3538, -0.6719].

Verdict: FALSIFIED_STRONGER_GLOSS. The verdict does not falsify the paper's
qualified claim that MDS is better in general; it rejects the stronger
every-comparator interpretation.

## Claim 6 — 1024-dimensional Cryo-EM

Claim contract: the Cryo-EM setup achieves the paper's displayed large RMSE
reductions at the reported scale.

Producer routes:

- reproduction/campaign/cryo_full_reproduction.py performs exact-scale
  retraining with 15,000 training datasets, 100 tests, 100 32-by-32 images,
  20 states, 2,000 posterior samples, and 1,024 RFFs.
- reproduction/campaign/cryo_full_reproduction.py and run_cryo_claim.py
  preserve the primary route.
- The discrete-state route checks all 20 posterior states.
- cryo_figure4_reconstruction.py and run_cryo_figure4.py reconstruct the
  displayed vector source values.
- cryo_falsification_audit.py and verify_cryo_falsification.py test whether
  alternative seeds, one worst query, or changed algorithm/noise models are
  valid counterexamples.

Evidence path:

- The claim-6 OpenResearch artifact contains the route notes, methods, source
  audit, and limitations.
- The existing Hugging Face page current-cryo-em contains the evaluator-facing
  route matrix and linked raw evidence.

Result: the paper's vector Figure 4 reports 58–63% reductions at epsilon
0.2–0.5. One exact-scale retraining observed 2.8–8.5% and worse clean
performance. The vector reconstruction is source evidence, not an independent
rerun; the other routes do not provide a valid universal falsification.

Verdict: BLOCKED_REPRODUCTION_REQUIRED. The unblocker is the authors' exact
saved checkpoint and Figure 4 training/test data realization, or an explicit
universal quantifier over retraining realizations.

## Reading rule

Verified means the stated scoped check passed. Falsified means the exact
universal or imported stronger contract has a checked counterexample.
Blocked means the evidence gap prevents a defensible verdict. None of these
labels should be generalized beyond the contract and limitations written
above.
