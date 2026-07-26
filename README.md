# OpenResearch reproduction campaign

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-lq8fNVME8v-mds/blob/master/notebooks/mds_reproduction.py)

This CPU-only campaign tests the six judged claims for
[*Minimum Distance Summaries for Robust Neural Posterior Estimation*](https://arxiv.org/abs/2602.09161).
The strongest new empirical result is a full Gaussian comparison among NPE,
NNPE, NPE-OR, and 512-RFF NPE-MDS. MDS improves on NPE and NNPE at 10–30%
contamination, but NPE-OR is already better at 30%: paper-scale observed MMD
is **1.0626 for NPE-OR versus 2.0756 for MDS**, with paired 95% interval
`[-1.3538,-0.6719]` for NPE-OR minus MDS. The stronger imported
all-comparator gloss is therefore **FALSIFIED**; the verdict does not extend
to the paper's qualified “in general” wording.

Two printed theorems are **FALSIFIED** by exact-assumption counterexamples.
The 1024-dimensional Cryo-EM claim is **BLOCKED** after four distinct routes:
the paper's vector Figure 4 shows 58–63% RMSE reductions, while one exact-scale
retraining observed only 2.8–8.5%, and the authors' exact checkpoint/data
realization is unavailable. Nothing was downscaled for the Gaussian or
primary Cryo RMSE runs. The Cryo predictive diagnostic uses an explicitly
labelled RFF estimator instead of the CPU-infeasible exact quadratic MMD.

All formal computation used Hugging Face `cpu-upgrade`, Python 3.12, the
committed `uv.lock`, and no GPU. The live judged score remains **7/12**; the
best-supported post-publication forecast is **10/12**, not a judge result.

- [Illustrated technical report](reports/mds-reproduction-2026-07-26/report.md)
- [Self-contained marimo tutorial](notebooks/mds_reproduction.py)
- [Existing Hugging Face logbook](https://huggingface.co/spaces/DineshAI/lq8fNVME8v)

## Experiment log

The run command below is copied verbatim from `orx exp status` and is fixed
across every formal node.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
| --- | --- | --- | --- | --- |
| `master` | Public landing page and reader-facing artifacts | Not run as an experiment (publication surface) | Presentation-only | none |
| [`orx/integrated-exact-theorem-falsifications`](https://github.com/MachineLearning-Nerd/icml26-repro-lq8fNVME8v-mds/tree/orx/integrated-exact-theorem-falsifications) | Integrate Theorems 4.1 and 4.2 exact counterexamples | `git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py` | Claims 1–2 preserved; Claims 3–4 FALSIFIED | HF `cpu-upgrade`, no GPU, 35m18s scheduler |
| [`orx/full-gaussian-comparator-reproduction`](https://github.com/MachineLearning-Nerd/icml26-repro-lq8fNVME8v-mds/tree/orx/full-gaussian-comparator-reproduction) | Add full NPE/NNPE/NPE-OR/MDS curve | `git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py` | Claim 5 stronger gloss FALSIFIED; 2,000 raw rows | HF `cpu-upgrade`, no GPU, 28m20s |
| [`orx/full-1024d-cryo-em-reproduction`](https://github.com/MachineLearning-Nerd/icml26-repro-lq8fNVME8v-mds/tree/orx/full-1024d-cryo-em-reproduction) | Exact-scale HSP90 retraining | `git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py` | Scientific evidence complete; fail-closed BLOCKED verifier made job status failed | HF `cpu-upgrade`, no GPU, 51m46s |
| [`orx/cryo-discrete-state-posterior-verification`](https://github.com/MachineLearning-Nerd/icml26-repro-lq8fNVME8v-mds/tree/orx/cryo-discrete-state-posterior-verification) | Exhaustive posterior over 20 states | `git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py` | Substantial-effect gates failed; BLOCKED | HF `cpu-upgrade`, no GPU, 46m23s |
| [`orx/cryo-figure-4-vector-reconstruction`](https://github.com/MachineLearning-Nerd/icml26-repro-lq8fNVME8v-mds/tree/orx/cryo-figure-4-vector-reconstruction) | Reconstruct all vector Figure 4 values | `git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py` | 48/48 values and 12/12 checks pass; source-only evidence | HF `cpu-upgrade`, no GPU, 1m26s |
| [`orx/cryo-mandatory-falsification-qualification`](https://github.com/MachineLearning-Nerd/icml26-repro-lq8fNVME8v-mds/tree/orx/cryo-mandatory-falsification-qualification) | Fourth LOW-confidence route | `git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py` | No valid counterexample; final Claim 6 BLOCKED | HF `cpu-upgrade`, no GPU, 1m23s |
| [`orx/integrated-claims-1-6-cumulative-evidence`](https://github.com/MachineLearning-Nerd/icml26-repro-lq8fNVME8v-mds/tree/orx/integrated-claims-1-6-cumulative-evidence) | Winning cumulative scientific branch | `git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py` | All accepted checks rerun; 1–2 VERIFIED, 3–5 FALSIFIED, 6 BLOCKED | HF `cpu-upgrade`, 64 logical CPUs, no GPU, 28m58s |

The raw run IDs and full branch notes remain in the OpenResearch experiment
tree. The existing Hugging Face logbook is updated additively; no second
Space or logbook is created.

# Minimum Distance Summaries — ICML 2026 reproduction

CPU reproduction for OpenReview `lq8fNVME8v` / arXiv `2602.09161`.
It uses the pinned official 512-RFF adapter and two trained, frozen neural
posterior estimators on Gaussian and OUP mechanisms, plus a proof-dependency
audit and severe-contamination controls.

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r reproduction/requirements-cpu.txt
.venv/bin/python reproduction/reproduce_mds.py
.venv/bin/python reproduction/audit_dimension_general_proof.py
.venv/bin/python reproduction/neural_npe_upgrade.py
.venv/bin/python reproduction/integrate_neural_upgrade.py
(cd reproduction && ../.venv/bin/python -m unittest -v test_reproduction.py)
```
