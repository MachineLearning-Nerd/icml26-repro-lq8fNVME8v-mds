# Claim 5 method

The formal run trains the pinned authors' standard NPE and NNPE from scratch
on 50,000 bivariate Gaussian datasets. Each contains 100 observations. It then
fits the official learned neural summary, official RFF-MDS adapter with 512
random features, and official OC-SVM removal/resampling baseline.

For each of 100 independently generated test datasets at each epsilon in
`{0.0, 0.1, 0.2, 0.3, 0.5}`, every method produces 2,000 posterior samples.
The primary result is the authors' biased, five-bandwidth MMD to 2,000 samples
from the analytic posterior conditional on the uncontaminated dataset. RMSE is
recorded as a secondary metric. All 2,000 method-by-dataset rows are printed
to the run log and written to CSV.

The paired unit is a test dataset. Ten thousand deterministic paired bootstrap
replicates estimate 95% intervals for each comparator-minus-MDS mean
difference. No paper curve, expected effect size, or formula chooses the
sample count or threshold after results are observed.

The independent checker reconstructs means directly from raw CSV, checks the
complete factorial design, verifies exact row count and hashes, audits paper
dimensions/settings, and proves that both neural estimators remained
bit-identical during test-time adaptation. The fail-closed verifier applies
the preregistered contract. A disabled-adaptation control must exit nonzero.

## Exact fixed command

```text
git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py
```

The environment is the repository-level `.venv` resolved by `uv.lock`.

## Compute estimate

The experiment is expected to use at most six CPU cores and has uncertain
runtime, conservatively estimated at 4–12 hours. It is therefore submitted to
Hugging Face `cpu-upgrade`; it is never executed locally. The program records
the actual logical CPU allocation and wall time in `runtime.json`.

## Memory-preserving implementation detail

Applying 512 RFFs to all 4.75 million fit observations at once would create a
roughly 19.5 GB float64 intermediate. The implementation processes 256
datasets per block, using the same fitted `RBFSampler`, seed, observations,
features, and per-dataset mean as the official implementation. This changes
only temporary materialization, not the estimator.
