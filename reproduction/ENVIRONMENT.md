# CPU environment

The reference protocol used:

```bash
python3.13 -m venv .venv-mds
.venv-mds/bin/python -m pip install -r reproduction/requirements-cpu.txt
```

This run uses the equivalent isolated Python 3.12 environment at
`.venv/bin/python` with the exact dependency versions in
`requirements-cpu.txt`. CUDA is disabled, all experiments run on CPU, and the
small neural/RFF workloads pin PyTorch intra-op and inter-op execution to one
thread to avoid thread-pool latency.
