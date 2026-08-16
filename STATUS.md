# STATUS — Minimum Distance Summaries

Updated 2026-08-16.

| Field | State |
| --- | --- |
| Repository | https://github.com/MachineLearning-Nerd/icml26-minimum-distance-summaries |
| Paper | Minimum Distance Summaries for Robust Neural Posterior Estimation |
| OpenReview | lq8fNVME8v |
| arXiv | 2602.09161v2 |
| Collection state | VERIFIED_SCOPED_WITH_FALSIFIED_THEOREMS_AND_BLOCKED_CRYO_CLAIM |
| Last recorded live judge score | 7/12 |
| Forecast in release artifacts | 9–10/12, forecast only |
| Formal compute | Hugging Face cpu-upgrade, CPU only, no GPU |

## Claim state

- Claim 1: VERIFIED_SCOPED — frozen NPE and test-time summary separation.
- Claim 2: VERIFIED_SCOPED — pinned 512-RFF, MSE, and L-BFGS path.
- Claim 3: FALSIFIED_AS_WRITTEN — Theorem 4.1 exact-assumption counterexample.
- Claim 4: FALSIFIED_AS_WRITTEN — Theorem 4.2 exact-assumption counterexample.
- Claim 5: FALSIFIED_STRONGER_GLOSS — NPE-OR beats MDS at 30% contamination;
  this is scoped to the stronger imported all-comparator wording.
- Claim 6: BLOCKED_REPRODUCTION_REQUIRED — exact Cryo-EM checkpoint/data
  realization is unavailable.

## Source and history state

The paper PDF and TeX are committed with hashes in
[SOURCE_MANIFEST.md](SOURCE_MANIFEST.md). The official implementation and
cryoSBI are git submodules pinned to exact commits. All reachable Git commits
in the normalized collection use the canonical identity
MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>.

The detailed evidence remains in the committed outputs, OpenResearch
artifacts, and the existing [Hugging Face logbook](https://huggingface.co/spaces/DineshAI/lq8fNVME8v).
See [CLAIM_EVIDENCE.md](CLAIM_EVIDENCE.md) before interpreting any result.

Thank you to the authors for making this work available for transparent
reproduction and for giving future readers a clear foundation to audit.
