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
