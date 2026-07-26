# Method

Load and execute the pinned official `rff.py` path, then audit seven named
mechanisms and measure the offline fit and query optimizer on Gaussian and
OUP. Compare the RFF objective against exact MMD and preserve the epsilon 0.5
single-start failure boundary.

Fixed command:
`git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py`.
