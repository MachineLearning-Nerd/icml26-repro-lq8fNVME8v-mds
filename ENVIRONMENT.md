# Reproduction environment

## Recorded formal environment

The formal OpenResearch runs used:

- Python 3.12 in ghcr.io/astral-sh/uv:python3.12-bookworm-slim.
- Hugging Face cpu-upgrade compute.
- No GPU.
- The committed uv.lock.
- Six estimated useful CPU cores, with the host exposing 64 logical CPUs.

The exact launch and subcommand ledger is in
[.openresearch/release/command_log.md](.openresearch/release/command_log.md).
The cumulative release used:

~~~text
git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py
~~~

The recorded cumulative run took 1,709.112072 seconds. Across the 14 jobs
through the scientific winner, the scheduler recorded 16,418 seconds and no
GPU use.

## Local setup

Initialize the two git submodules first:

~~~bash
git submodule update --init --recursive
uv sync --frozen
~~~

The project requires Python 3.12 according to pyproject.toml. The small
requirements file documents the pinned CPU scientific packages; uv.lock is
the authoritative complete environment for the campaign.

## Commands

Run the complete recorded campaign:

~~~bash
uv run python reproduction/run_all.py
~~~

Run the cumulative regression suite after submodules are initialized:

~~~bash
uv run python -m unittest -v reproduction/test_reproduction.py
~~~

Run the lightweight source and release checks:

~~~bash
python3 verify_final.py
uv run python reproduction/validate_release.py
~~~

The full campaign is intentionally not run as part of a documentation-only
checkout verification. The committed outputs and OpenResearch artifacts are
the evidence for the recorded run; a new run must produce a new receipt and
should not overwrite the historical evidence without an explicit reason.

## Reproduction boundaries

- The Gaussian core uses 100 observations per dataset, 2,000 training
  datasets, 50 trials per contamination level, and 512 RFFs.
- The neural upgrade contains frozen Gaussian and OUP posterior artifacts.
- Cryo-EM routes are computationally and scientifically separate from the
  Gaussian core.
- CUDA is disabled by the producer scripts.
- Cross-machine wall-clock values are reported as observed envelopes, not
  universal performance guarantees.
