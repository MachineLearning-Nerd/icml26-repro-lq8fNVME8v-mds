# Minimum Distance Summaries for Robust Neural Posterior Estimation — ICML 2026 reproduction

This is a paper-first reproduction and audit of [Minimum Distance Summaries for
Robust Neural Posterior Estimation](https://arxiv.org/abs/2602.09161). It keeps
the paper, the pinned official implementation, executable reproduction code,
committed outputs, claim contracts, independent checkers, controls, and
limitations together so that a reader can distinguish a verified result from
an unresolved or falsified claim.

The repository is maintained by [MachineLearning-Nerd](https://github.com/MachineLearning-Nerd).
The previous repository name was
icml26-repro-lq8fNVME8v-mds; the canonical home is now
https://github.com/MachineLearning-Nerd/icml26-minimum-distance-summaries.

## Paper and implementation

| Field | Value |
| --- | --- |
| Paper | Minimum Distance Summaries for Robust Neural Posterior Estimation |
| Authors | Sherman Khoo, Dennis Prangle, Song Liu, and Mark Beaumont |
| Venue | Proceedings of the 43rd International Conference on Machine Learning, 2026 |
| arXiv | [2602.09161v2](https://arxiv.org/abs/2602.09161) |
| OpenReview | [lq8fNVME8v](https://openreview.net/forum?id=lq8fNVME8v) |
| Official implementation | [Shermjj/Minimum-Distance-Summaries](https://github.com/Shermjj/Minimum-Distance-Summaries) |
| Pinned official source | Git submodule commit 45158124f0cbdc2f6c1ac602c9fc5501dce20af3 |
| External Cryo-EM dependency | [flatironinstitute/cryoSBI](https://github.com/flatironinstitute/cryoSBI), commit 8e5832ecda626e9ab58d18cb215b5db6789533ee |

## Current status

The last recorded live judge score is 7/12 at the published
[Hugging Face logbook](https://huggingface.co/spaces/DineshAI/lq8fNVME8v).
The 9–10/12 value mentioned in the release artifacts is a forecast of what a
future evaluator might award, not a new judge result.

The current independent audit is:

- Claims 1 and 2: VERIFIED_SCOPED.
- Theorem 4.1 (Claim 3): FALSIFIED_AS_WRITTEN by an exact-assumption
  counterexample.
- Theorem 4.2 (Claim 4): FALSIFIED_AS_WRITTEN by an exact-assumption
  counterexample.
- The stronger imported all-comparator Gaussian gloss (Claim 5):
  FALSIFIED_STRONGER_GLOSS. This does not reject the paper's qualified
  wording that MDS is better in general.
- The 1024-dimensional Cryo-EM claim (Claim 6):
  BLOCKED_REPRODUCTION_REQUIRED. The paper's exact checkpoint and data
  realization are unavailable, and an exact-scale retraining disagrees with
  the displayed result.

## What the paper is doing

The paper proposes minimum-distance summaries (MDS) as a post-hoc,
test-time adaptation layer for a pretrained neural posterior estimator (NPE).
For a new observation, MDS optimizes the summary statistic so that the
simulator-conditioned predictive distribution is close to the observation
under maximum mean discrepancy (MMD). The adapted summary is then passed to the
frozen NPE. The intended separation is important: the expensive posterior
training is offline, while the robust query adaptation is modular and
amortized.

The implementation path audited here uses a Gaussian-kernel random Fourier
feature approximation, an MSE-trained conditional embedding regressor, and
strong-Wolfe L-BFGS at test time. The paper specifies 512 RFFs; the official
repository defaults to 256, so this reproduction explicitly overrides the
dimension to 512 and records the discrepancy.

## Claim ledger

| ID | Paper or audit claim | Verdict | Evidence boundary |
| --- | --- | --- | --- |
| 1 | MDS changes the query summary while the pretrained NPE remains frozen. | VERIFIED_SCOPED | Frozen posterior hashes and 550 paired adaptations pass the cumulative checks. |
| 2 | The RFF/MSE/L-BFGS implementation is a lightweight test-time adapter. | VERIFIED_SCOPED | Official pinned RFF path, signatures, CPU timing, and approximation checks pass. |
| 3 | Under Appendix A.1 assumptions, the Theorem 4.1 posterior-KL influence bound is finite. | FALSIFIED_AS_WRITTEN | All ten printed assumptions pass in a constructed case, but the target objective has a quartically flat minimum; KL divided by contamination grows from 20.87 at 1e-3 to 4547.80 at 1e-10. |
| 4 | Under Appendix A.3 assumptions, original-summary posterior consistency implies MDS-summary consistency. | FALSIFIED_AS_WRITTEN | An exact conditional counterexample satisfies the printed assumptions, while MDS selects an escaping point mass; the missing topology is stronger than bounded ISPD alone. |
| 5 | Imported stronger gloss: MDS beats every named comparator below 40% contamination. | FALSIFIED_STRONGER_GLOSS | In the full four-method Gaussian run, NPE-OR beats MDS at epsilon 0.3: 1.0626 versus 2.0756, paired 95% interval [-1.3538, -0.6719]. |
| 6 | The 1024-dimensional Cryo-EM experiment achieves the paper's large RMSE reductions. | BLOCKED_REPRODUCTION_REQUIRED | The paper displays 58–63% reductions; exact-scale retraining observed 2.8–8.5%. Four routes cannot adjudicate the difference without the authors' exact checkpoint/data realization. |

The machine-readable version of this ledger is in [claims.json](claims.json).
The claim-to-evidence paths and controls are expanded in
[CLAIM_EVIDENCE.md](CLAIM_EVIDENCE.md).

## How each claim is produced

Every claim follows the same audit path:

1. Pin the paper PDF, source TeX, official implementation, and external
   dependency.
2. Define the claim contract before reading the result.
3. Run the narrow producer for the claim and preserve raw outputs.
4. Recompute summaries with an independent checker.
5. Run a negative or scope control where the claim needs one.
6. Record the verdict together with its exact limitation.

| Claim | Producer code | Committed evidence | Independent check and control |
| --- | --- | --- | --- |
| 1 | reproduction/reproduce_mds.py and reproduction/neural_npe_upgrade.py | outputs/full/neural_upgrade_summary.json, neural_gaussian_aggregate.csv, neural_oup_aggregate.csv | reproduction/test_reproduction.py and .openresearch/artifacts/claim_1_plugin_separation |
| 2 | reproduction/reproduce_mds.py | outputs/full/source_and_code_audit.json and experiment_summary.json | Official-code signature audit, exact-MMD comparison, and .openresearch/artifacts/claim_2_rff_implementation |
| 3 | reproduction/campaign/theorem_4_1_counterexample.py and run_theorem_4_1.py | .openresearch/artifacts/claim_3_theorem_4_1/formal_results.json | formal_independent_checker_output.json plus the correctly specified control |
| 4 | reproduction/campaign/theorem_4_2_counterexample.py and run_theorem_4_2.py | .openresearch/artifacts/claim_4_theorem_4_2/formal_result.json and formal_results.csv | formal_independent_checker_output.json plus the Gaussian-C0 control |
| 5 | reproduction/campaign/gaussian_full_comparators.py and run_gaussian_comparators.py | outputs/full/gaussian_trials.csv, gaussian_aggregate.csv, and the claim-5 raw CSV | 11 integrity checks, disabled-adaptation control, and verifier_output.json |
| 6 | reproduction/campaign/cryo_full_reproduction.py, run_cryo_claim.py, and the other Cryo route scripts | .openresearch/artifacts/claim_6_cryo_em and the linked Hugging Face page | Exact-scale, discrete-state, vector-source, and mandatory-falsification routes; no defensible final verdict |

## Branch map

The public branch names now describe the experiment role. The complete
legacy-to-canonical mapping, purpose, producer, result, and branch policy are
in [BRANCH_AUDIT.md](BRANCH_AUDIT.md).

| Canonical branch | Role |
| --- | --- |
| main | Reader-facing landing page and final documentation |
| baseline/judged-7-of-12 | Frozen judged baseline and uv-lock environment |
| baseline/portable-cumulative | Portable CPU latency and cumulative baseline |
| audit/claim-3-theorem-4-1 | Exact Theorem 4.1 counterexample |
| audit/claim-4-theorem-4-2 | Exact Theorem 4.2 counterexample |
| proof/claims-3-4-integrated | Integrated exact theorem falsifications |
| experiment/gaussian-full-comparators | Full NPE, NNPE, NPE-OR, and MDS comparison |
| experiment/cryo-em-full-1024d | Exact-scale Cryo-EM retraining route |
| experiment/cryo-discrete-posterior | Exhaustive 20-state Cryo posterior route |
| audit/cryo-figure-4-reconstruction | Vector Figure 4 source reconstruction |
| audit/cryo-falsification-qualification | Qualification of possible Cryo falsifications |
| release/integrated-claims-1-6 | Cumulative scientific release |
| release/evaluator-visible | Evaluator-facing release surface |

No branch is presented as stronger evidence than its producer and controls
support. The cumulative release is the reader's main scientific entry point;
the individual branches preserve the reasoning path.

## Reproduce the recorded campaign

The repository uses Python 3.12 and the committed uv lock. Initialize both
source submodules before importing the official adapter:

~~~bash
git clone https://github.com/MachineLearning-Nerd/icml26-minimum-distance-summaries.git
cd icml26-minimum-distance-summaries
git submodule update --init --recursive
uv sync --frozen
uv run python reproduction/run_all.py
~~~

The full command is a multi-hour CPU campaign and is not required to inspect
the committed evidence. The lightweight release and source checks are
described in [ENVIRONMENT.md](ENVIRONMENT.md) and
[SOURCE_AUDIT.md](SOURCE_AUDIT.md). The recorded formal runs used the
Hugging Face cpu-upgrade flavor with no GPU; the exact command ledger is in
.openresearch/release/command_log.md.

## Important limitations

- The controlled Gaussian experiment reproduces the core RFF-MDS mechanism;
  it is not a rerun of every OUP, SIR, Cryo-EM, NPE-PFN, or comparator result in
  the paper.
- Theorem 4.1 and Theorem 4.2 verdicts are exact counterexamples to the
  universal statements under their printed assumptions. They are not claims
  that every practical MDS objective fails.
- The Cryo-EM result is intentionally blocked rather than promoted to a
  falsification. The missing author checkpoint and exact data realization are
  material evidence gaps.
- Historical judge scores and forecasts are kept separate. A forecast in an
  artifact is never reported as a new evaluation result.

## Citation

~~~bibtex
@inproceedings{khoo2026minimum,
  title     = {Minimum Distance Summaries for Robust Neural Posterior Estimation},
  author    = {Khoo, Sherman and Prangle, Dennis and Liu, Song and Beaumont, Mark},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  eprint    = {2602.09161},
  archivePrefix = {arXiv},
  primaryClass = {stat.ML},
  url       = {https://arxiv.org/abs/2602.09161}
}
~~~

For software citation metadata, see [CITATION.cff](CITATION.cff).

## Thank you

Thank you to Sherman Khoo, Dennis Prangle, Song Liu, and Mark Beaumont for
making the paper, source, and scientific method available for careful
reproduction. This audit is intended as a transparent companion to the
paper: it preserves positive results, exposes scope boundaries, and records
unresolved discrepancies so future readers and the authors can build on a
reproducible evidence trail.

## Repository guide

- [STATUS.md](STATUS.md) — current one-page status.
- [CLAIM_EVIDENCE.md](CLAIM_EVIDENCE.md) — claim contracts, producers,
  evidence, controls, and limitations.
- [BRANCH_AUDIT.md](BRANCH_AUDIT.md) — canonical branch lineage.
- [SOURCE_AUDIT.md](SOURCE_AUDIT.md) — paper/code provenance and source
  boundaries.
- [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md) — pinned source hashes.
- [ENVIRONMENT.md](ENVIRONMENT.md) — setup and recorded compute protocol.
- [reports/mds-reproduction-2026-07-26/report.md](reports/mds-reproduction-2026-07-26/report.md)
  — illustrated technical report.
- [notebooks/mds_reproduction.py](notebooks/mds_reproduction.py) — self-contained
  reader-oriented tutorial.
- [EVIDENCE_MANIFEST.json](EVIDENCE_MANIFEST.json) — hashes for the principal
  committed evidence files.
