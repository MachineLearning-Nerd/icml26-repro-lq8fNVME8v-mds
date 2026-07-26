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

## Route 2: complete discrete-state posterior

The simulator parameter and prior are exactly discrete over the 20 HSP90
states `{0,...,19}`. The first route followed the authors' implementation and
sampled the continuous spline-flow density. Route 2 additionally evaluates
that same frozen learned density at every one of the 20 admissible states and
normalizes the 20 density values. Posterior mean, RMSE, predictive RFF
distance, entropy, and weight normalization are then reconstructed from the
complete finite domain. No state is omitted and no Monte Carlo posterior
sample is used for this route.

Both interpretations are emitted in every raw trial row. The grid route is an
independent domain-faithful reading of the discrete task, not a claim that the
authors' Figure 4 runner used grid normalization. The same preregistered effect
and clean-regime thresholds are applied to the grid metrics.

## Route 3: vector reconstruction of Figure 4

The exact paper PDF page 8 is converted once to a committed SVG. The
conversion retains the vector plotting paths. A standard-library parser finds
the eight six-point curves and their 48 vertical uncertainty intervals by
exact color, transform, line width, and epsilon-coordinate grid. It fits each
y-axis calibration independently from every visible labeled tick:

- RMSE ticks 2 through 8 at seven vector coordinates.
- Predictive-MMD ticks 0.032 through 0.038 at four vector coordinates.

No curve value is manually entered. An independent checker uses separately
hard-coded source hashes, axis coefficients, epsilon grids, and the full
factorial contract to reconstruct the CSV. A disabled-adaptation curve control
must exit nonzero.

This route can rigorously establish the values displayed by Figure 4 and
compare them with the full-scale experimental routes. It cannot independently
verify the underlying experiment because the figure is part of the paper
being tested. The route therefore remains fail-closed with verdict BLOCKED.

## Route 4: mandatory falsification qualification

The exact 1,200 Route 1 trial rows are reconstructed from formal run
`1df5f0be-40f8-495f-bf0c-ea9ada0ebc9f`. Re-serializing the logged rows with
the original CSV schema reproduces the formal raw SHA-256 exactly:
`ef27c8f5d79c1cc225d1fba328c3f4de86f0b3bf914b7da1de47797dac3150de`.

The audit first restates the claim, HSP90 domain, contamination, scale,
estimand, and source quantifier. It then evaluates three materially distinct
candidate counterexamples:

1. The full-scale independent seed-42 retraining, which satisfies the stated
   task assumptions but is a different realized experiment.
2. The worst valid individual query, which does not contradict a 100-test
   aggregate report.
3. Disabled MDS or altered noise, which violates the named algorithm or
   contamination.

A candidate qualifies only if it satisfies the paper assumptions and
contradicts the exact quantified statement. The paper ties the claim to a
finite Figure 4 experiment and does not universally quantify over every
training seed, realization, test set, or query. Consequently, a rigorous
reproduction discrepancy alone is not promoted to FALSIFIED. The negative
control deliberately attempts that invalid promotion and must exit nonzero.

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
