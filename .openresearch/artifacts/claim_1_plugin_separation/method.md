# Method

Train the Gaussian and OUP NPEs once, hash every tensor, and run ordinary and
MDS-adapted queries against the same objects. Rehash after all adaptations.
`integrate_neural_upgrade.py` independently asserts architecture, parameter
count, paired-query count, and before/after tensor equality.

Fixed command:
`git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py`.
Seeds and exact environment are committed in the executable source and
`uv.lock`.
