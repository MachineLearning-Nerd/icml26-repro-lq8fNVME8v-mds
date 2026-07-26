# Claim 6 method

The formal run generates all 15,000 HSP90 training datasets from the pinned
cryoSBI simulator. Each dataset contains 100 independent 32×32 images from one
of 20 conformational states. The official six-dimensional summary is computed
from per-image skewness, kurtosis, and intensity range.

A neural spline-flow NPE is trained from scratch with the paper configuration:
learning rate 5e-4, batch size 256, at most 500 epochs, and 20-epoch early
stopping. The task-configured 1,024-RFF MDS adapter is fitted on the first
10,000 datasets, with 5% calibration, two 256-unit regressor layers, 100
regressor epochs, and the official L-BFGS test-time adaptation.

For each epsilon in `{0,.1,.2,.3,.4,.5}`, the same 100 clean test datasets are
contaminated by replacing exactly `epsilon × 100` images with normalized pure
Gaussian noise. NPE and MDS each produce 2,000 posterior samples. The primary
paper metric is RMSE against the true state. Ten thousand deterministic paired
bootstrap replicates quantify NPE-minus-MDS differences.

The secondary posterior-predictive diagnostic is the squared distance between
full-data 1,024-RFF mean embeddings. State-conditional embeddings use all one
million RFF-fit training images; clean embeddings use all 100 images in each
test dataset; posterior weights use all 2,000 samples. It is a faithful
full-data approximation under the same RBF/RFF kernel as MDS, but it is not
called the paper's exact quadratic five-bandwidth MMD.

## Memory-preserving equivalence

The first 10,000 raw training datasets are stored in a temporary 4.096 GB
NumPy memmap. RFFs are transformed eight datasets at a time. This changes
temporary materialization only: the same images, random features, median
heuristic, per-dataset means, summaries, split, and calibration are used.
Datasets 10,001–15,000 are fully simulated for NPE training and discarded
after their exact summaries are computed.

## Exact fixed command

```text
git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py
```

The environment is the repository-level `.venv` from `uv.lock`.

## Compute estimate

The dense 1024×1024 random-feature transform over one million images dominates
the workload. Estimated useful parallelism is 32 CPU cores, temporary disk is
about 4.1 GB, and runtime is uncertain at roughly 12–24 hours. The only
authorized target is Hugging Face `cpu-upgrade`, with a 24-hour timeout and no
GPU. Actual allocation and runtime are written to the evidence.
