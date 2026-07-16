# Reproduction protocol, artifacts, and hashes


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b8cf0fa90471", "created_at": "2026-07-16T15:52:03+00:00", "title": "Fail-closed verification and provenance"}
-->
# Exact reproduction protocol

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r reproduction/requirements-cpu.txt
.venv/bin/python reproduction/reproduce_mds.py
.venv/bin/python reproduction/audit_dimension_general_proof.py
.venv/bin/python reproduction/neural_npe_upgrade.py
.venv/bin/python reproduction/integrate_neural_upgrade.py
(cd reproduction && ../.venv/bin/python -m unittest -v test_reproduction.py)
```

The dedicated interpreter avoids ambient-`python3` mismatches. Exact successful dependencies are pinned in `requirements-cpu.txt`; `ENVIRONMENT.md` documents the workspace-root invocation.

Final verifier output: `PASS: 16/16 independent assertions; CPU-only bundle verified`.

The assertions cover exact paper identity/four-claim snapshot, primary-source hashes plus commit, seven official-code signatures, all original and neural trials, clean-gate preservation, effect-size/paired-statistic thresholds, millisecond timing, RFF-vs-exact agreement, severe-case disclosure, bounded influence, monotone posterior contraction, actual saved neural checkpoints, frozen NPE state, the paper-scale OUP test, and explicit theorem/package boundaries.

**Environment:** Linux-7.0.9-arch2-1-x86_64-with-glibc2.43; x86_64; Python `3.12.13`; NumPy `2.3.5`; SciPy `1.17.1`; scikit-learn `1.8.0`; PyTorch `2.12.0+cu130`; device `cpu`; GPU used `false`; reproduction wall time **20.016s**.

**Primary hashes:**

- PDF `1fc774ab166496d0861720b14212204c46cf920dc22c5df006d1d48f5eba01f0`
- arXiv TeX `ff81fd973e3bcba86fb23e9a0c102ec88e240f62361315c7875de54e29ea4fd2`
- official RFF module `0a7e8a2261c9ab9f8b327c01335dd7ff9cb7d5efd48abcf1980bf9c125d329b9`
- official commit `45158124f0cbdc2f6c1ac602c9fc5501dce20af3`

The artifact contains all 21 output files, including two neural checkpoints, raw neural trials/training curves, and the proof certificate. Full paired trials, consistency trials, source audit, environment, summaries, frozen protocol, exact requirements and verification scripts are present in the logged Trackio artifact.


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_6175e08b4c49", "created_at": "2026-07-16T15:52:04+00:00", "title": "Complete CPU reproduction workspace", "artifact": "minimum-distance-summaries-repro/minimum-distance-summaries-cpu-reproduction:v0", "artifact_type": "dataset"}
-->
**📦 Artifact** `minimum-distance-summaries-repro/minimum-distance-summaries-cpu-reproduction:v0` · dataset

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#minimum-distance-summaries-repro/minimum-distance-summaries-cpu-reproduction:v0


---
<!-- trackio-cell
{"type": "code", "id": "cell_b0f46534007a", "created_at": "2026-07-16T15:52:37+00:00", "title": "Core official-RFF reproduction", "command": [".venv/bin/python", "reproduction/reproduce_mds.py"], "exit_code": 0, "duration_s": 19.136}
-->
````bash
$ .venv/bin/python reproduction/reproduce_mds.py
````

exit 0 · 19.1s


````python title=reproduce_mds.py
#!/usr/bin/env python3
"""CPU-only reproduction of the four ICML challenge claims for MDS.

The empirical experiment deliberately exercises the authors' isolated RFF
adapter module, while the theorem checks use closed-form Gaussian/RBF special
cases.  No network, GPU, cloud job, or model API is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar
from scipy.stats import ttest_rel
import sklearn
import torch


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_RFF = ROOT / "source" / "official-repo" / "src" / "tt_sbi" / "tta" / "rff.py"
PAPER_PDF = ROOT / "paper" / "2602.09161.pdf"
PAPER_TEX = ROOT / "source" / "arxiv" / "arxiv_main.tex"
CLAIMS_PATH = ROOT / "claims.json"
PAPER_ID = "lq8fNVME8v"
TITLE = "Minimum Distance Summaries for Robust Neural Posterior Estimation"

TRAIN_SEED = 42
TEST_SEEDS = tuple(range(1000, 1050))
N_TRAIN = 2_000
N_OBS = 100
PRIOR_VAR = 4.0
CONTAMINATION_SHIFT = 8.0
EPSILONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_official_rff_module():
    spec = importlib.util.spec_from_file_location("mds_official_rff", OFFICIAL_RFF)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {OFFICIAL_RFF}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT / "source" / "official-repo"), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_audit(out: Path) -> dict:
    claims = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))[PAPER_ID]
    tex_lines = PAPER_TEX.read_text(encoding="utf-8").splitlines()

    def locate(needle: str) -> int:
        hits = [i + 1 for i, line in enumerate(tex_lines) if needle in line]
        if not hits:
            raise AssertionError(f"Missing source anchor: {needle}")
        return hits[0]

    official = OFFICIAL_RFF.read_text(encoding="utf-8")
    patterns = {
        "sklearn_RBFSampler": "RBFSampler" in official,
        "conditional_embedding_regression": "Z_train = self.rff.fit_transform" in official,
        "mse_training": "nn.MSELoss()" in official,
        "lbfgs_test_time": "torch.optim.LBFGS" in official,
        "strong_wolfe_line_search": "line_search_fn='strong_wolfe'" in official,
        "cpu_device_parameter": 'device: str = "cpu"' in official,
        "calibrated_gate": "_calibrate_tau_from_data" in official,
    }
    if not all(patterns.values()):
        raise AssertionError(f"Official-code audit failed: {patterns}")

    repo_root = ROOT / "source" / "official-repo"
    init_path = repo_root / "src" / "tt_sbi" / "tta" / "__init__.py"
    metrics_path = repo_root / "src" / "tt_sbi" / "utils" / "metrics.py"
    pyproject = repo_root / "pyproject.toml"
    packaging_caveat = {
        "package_import_observation": (
            "Importing tt_sbi.tta triggers ncpp -> utils.metrics -> sbibm, but sbibm "
            "is absent from pyproject dependencies. The reproduction therefore loads "
            "the self-contained official rff.py module directly."
        ),
        "tta_init_sha256": sha256(init_path),
        "metrics_sha256": sha256(metrics_path),
        "pyproject_sha256": sha256(pyproject),
        "paper_rff_dimension": 512,
        "repository_default_rff_dimension": 256,
        "reproduction_override": 512,
    }
    audit = {
        "paper_id": PAPER_ID,
        "title": TITLE,
        "arxiv_id": "2602.09161",
        "openreview_url": f"https://openreview.net/forum?id={PAPER_ID}",
        "challenge_claims_exact": claims,
        "primary_source": {
            "pdf_sha256": sha256(PAPER_PDF),
            "tex_sha256": sha256(PAPER_TEX),
            "official_repo_commit": git_output("rev-parse", "HEAD"),
            "official_rff_sha256": sha256(OFFICIAL_RFF),
        },
        "anchors_arxiv_main_tex": {
            "method_definition": locate("minimum-distance summary (MDS) as follows"),
            "rff_objective": locate("random Fourier feature objective"),
            "decoder_mean_embedding": locate("Amortized Decoder Mean Embedding"),
            "test_time_adaptation": locate("Test-Time MDS Adaptation"),
            "algorithm_1": locate("MMD minimum distance summary with SGD"),
            "theorem_4_1": locate("thm:robustness"),
            "theorem_4_2": locate("thm:consistency"),
            "public_implementation": locate("Our implementation is publicly available"),
            "paper_scale_rff_512": locate("fixed dimension of \\( 512 \\)"),
        },
        "official_code_patterns": patterns,
        "packaging_and_config_caveats": packaging_caveat,
    }
    write_json(out / "source_and_code_audit.json", audit)
    return audit


def exact_conditional_mmd_summary(x: np.ndarray, gamma: float, conditional_var: float) -> float:
    """Global scalar MMD minimizer for N(s, conditional_var) vs empirical x."""
    values = np.asarray(x, dtype=float).reshape(-1)
    denom = 1.0 + 2.0 * gamma * conditional_var

    def score(s: float) -> float:
        return float(np.exp(-gamma * (s - values) ** 2 / denom).mean() / math.sqrt(denom))

    lo, hi = float(values.min() - 2.0), float(values.max() + 2.0)
    grid = np.linspace(lo, hi, 1201)
    # Chunked/vectorized grid search avoids relying on a local basin.
    scores = np.empty_like(grid)
    for start in range(0, len(grid), 200):
        g = grid[start : start + 200, None]
        scores[start : start + 200] = np.exp(-gamma * (g - values[None, :]) ** 2 / denom).mean(axis=1)
    j = int(np.argmax(scores))
    left = grid[max(0, j - 2)]
    right = grid[min(len(grid) - 1, j + 2)]
    if left == right:
        return float(grid[j])
    result = minimize_scalar(lambda s: -score(float(s)), bounds=(left, right), method="bounded")
    if not result.success:
        raise RuntimeError(f"Exact MMD scalar optimization failed: {result.message}")
    return float(result.x)


def paired_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def run_official_rff_experiment(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    mod = load_official_rff_module()
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    rng = np.random.default_rng(TRAIN_SEED)
    theta = rng.normal(0.0, math.sqrt(PRIOR_VAR), size=(N_TRAIN, 1))
    x_train = theta[:, None, :] + rng.normal(size=(N_TRAIN, N_OBS, 1))
    s_train = x_train.mean(axis=1)

    cfg = mod.RFFTTAConfig(
        rff_dim=512,
        hidden_dims=[256, 256],
        regressor_epochs=50,
        batch_size=256,
        tta_steps=200,
        seed=TRAIN_SEED,
    )
    adapter = mod.RFFTTAAdapter(cfg, device="cpu")
    fit_start = time.perf_counter()
    adapter.fit(x_train, s_train, calibration_split=0.05, alpha=0.05)
    fit_seconds = time.perf_counter() - fit_start

    rows: list[dict] = []
    post_var = 1.0 / (1.0 / PRIOR_VAR + N_OBS)
    post_factor = post_var * N_OBS
    conditional_var = 1.0 - 1.0 / N_OBS

    for eps in EPSILONS:
        n_outliers = int(round(N_OBS * eps))
        for test_seed in TEST_SEEDS:
            test_rng = np.random.default_rng(test_seed)
            clean = test_rng.normal(0.0, 1.0, size=(N_OBS, 1))
            outliers = test_rng.normal(CONTAMINATION_SHIFT, 1.0, size=(N_OBS, 1))
            observed = clean.copy()
            observed[:n_outliers] = outliers[:n_outliers]
            # Order must not matter for a permutation-invariant summary/embedding.
            observed = observed[test_rng.permutation(N_OBS)]
            s_observed = float(observed.mean())
            s_tensor = torch.tensor([s_observed], dtype=torch.float32)

            start = time.perf_counter()
            result = adapter.adapt(s_tensor, observed)
            adapt_ms = (time.perf_counter() - start) * 1_000.0
            s_mds = float(result["best_s"].reshape(-1)[0])
            s_exact = exact_conditional_mmd_summary(observed, float(adapter.gamma), conditional_var)
            s_median = float(np.median(observed))
            s_oracle = float(clean.mean())

            p_base = post_factor * s_observed
            p_mds = post_factor * s_mds
            p_exact = post_factor * s_exact
            p_median = post_factor * s_median
            p_oracle = post_factor * s_oracle

            losses = [float(z) for z in result["losses"]]
            rows.append(
                {
                    "epsilon": eps,
                    "seed": test_seed,
                    "n_obs": N_OBS,
                    "n_outliers": n_outliers,
                    "shift": CONTAMINATION_SHIFT,
                    "s_observed": s_observed,
                    "s_mds_rff": s_mds,
                    "s_mds_exact": s_exact,
                    "s_sample_median": s_median,
                    "s_oracle_clean": s_oracle,
                    "posterior_mean_baseline": p_base,
                    "posterior_mean_mds_rff": p_mds,
                    "posterior_mean_mds_exact": p_exact,
                    "posterior_mean_median": p_median,
                    "posterior_mean_oracle": p_oracle,
                    "sq_error_baseline": p_base**2,
                    "sq_error_mds_rff": p_mds**2,
                    "sq_error_mds_exact": p_exact**2,
                    "sq_error_median": p_median**2,
                    "abs_error_baseline": abs(p_base),
                    "abs_error_mds_rff": abs(p_mds),
                    "rff_exact_abs_summary_gap": abs(s_mds - s_exact),
                    "gate_passed": bool(result["gate_passed"]),
                    "lbfgs_evaluations": int(result["n_steps"]),
                    "objective_initial": losses[0],
                    "objective_final": losses[-1],
                    "objective_reduction": losses[0] - losses[-1],
                    "adapt_ms": adapt_ms,
                }
            )

    trials = pd.DataFrame(rows)
    trials.to_csv(out / "gaussian_trials.csv", index=False, float_format="%.12g")

    agg_rows: list[dict] = []
    boot_rng = np.random.default_rng(20260715)
    for eps, group in trials.groupby("epsilon", sort=True):
        base_sq = group["sq_error_baseline"].to_numpy()
        mds_sq = group["sq_error_mds_rff"].to_numpy()
        delta_abs = group["abs_error_baseline"].to_numpy() - group["abs_error_mds_rff"].to_numpy()
        ci_lo, ci_hi = paired_ci(delta_abs, boot_rng)
        base_rmse = float(np.sqrt(base_sq.mean()))
        mds_rmse = float(np.sqrt(mds_sq.mean()))
        exact_rmse = float(np.sqrt(group["sq_error_mds_exact"].mean()))
        median_rmse = float(np.sqrt(group["sq_error_median"].mean()))
        if np.allclose(delta_abs, 0.0):
            pvalue = 1.0
        else:
            pvalue = float(ttest_rel(group["abs_error_baseline"], group["abs_error_mds_rff"]).pvalue)
        agg_rows.append(
            {
                "epsilon": eps,
                "trials": len(group),
                "baseline_posterior_rmse": base_rmse,
                "mds_rff_posterior_rmse": mds_rmse,
                "mds_exact_posterior_rmse": exact_rmse,
                "sample_median_posterior_rmse": median_rmse,
                "rmse_reduction_pct": 100.0 * (base_rmse - mds_rmse) / base_rmse if base_rmse else 0.0,
                "paired_abs_error_improvement_mean": float(delta_abs.mean()),
                "paired_abs_error_improvement_ci95_low": ci_lo,
                "paired_abs_error_improvement_ci95_high": ci_hi,
                "paired_ttest_pvalue": pvalue,
                "mds_win_rate": float((group["abs_error_mds_rff"] < group["abs_error_baseline"]).mean()),
                "gate_pass_rate": float(group["gate_passed"].mean()),
                "mean_lbfgs_evaluations": float(group["lbfgs_evaluations"].mean()),
                "median_adapt_ms": float(group["adapt_ms"].median()),
                "p95_adapt_ms": float(group["adapt_ms"].quantile(0.95)),
                "mean_rff_exact_abs_summary_gap": float(group["rff_exact_abs_summary_gap"].mean()),
                "mean_objective_reduction": float(group["objective_reduction"].mean()),
            }
        )
    aggregate = pd.DataFrame(agg_rows)
    aggregate.to_csv(out / "gaussian_aggregate.csv", index=False, float_format="%.12g")

    contaminated = aggregate.query("epsilon >= 0.1 and epsilon <= 0.4")
    runtime_nonzero = trials.query("epsilon > 0")["adapt_ms"]
    experiment_summary = {
        "design": {
            "official_module": str(OFFICIAL_RFF.relative_to(ROOT)),
            "device": "cpu",
            "train_seed": TRAIN_SEED,
            "test_seeds": [TEST_SEEDS[0], TEST_SEEDS[-1]],
            "test_trials_per_level": len(TEST_SEEDS),
            "n_train_datasets": N_TRAIN,
            "n_observations_per_dataset": N_OBS,
            "prior_variance": PRIOR_VAR,
            "contamination": "one-sided Normal(theta+8, 1) replacing epsilon*N observations",
            "epsilon_levels": list(EPSILONS),
            "rff_dimension": 512,
            "regressor_hidden_dims": [256, 256],
            "regressor_epochs": 50,
            "calibration_fraction": 0.05,
            "calibration_alpha": 0.05,
            "lbfgs_max_steps": 200,
        },
        "offline_fit_seconds": fit_seconds,
        "final_regressor_mse": float(adapter.regressor_losses[-1]),
        "calibrated_tau": float(adapter.calibrated_tau),
        "adapter_gamma": float(adapter.gamma),
        "headline": {
            "mean_rmse_reduction_pct_eps_0p1_to_0p4": float(contaminated["rmse_reduction_pct"].mean()),
            "minimum_rmse_reduction_pct_eps_0p1_to_0p4": float(contaminated["rmse_reduction_pct"].min()),
            "minimum_win_rate_eps_0p1_to_0p4": float(contaminated["mds_win_rate"].min()),
            "clean_gate_pass_rate": float(aggregate.loc[aggregate.epsilon == 0.0, "gate_pass_rate"].iloc[0]),
            "clean_rmse_delta": float(
                aggregate.loc[aggregate.epsilon == 0.0, "mds_rff_posterior_rmse"].iloc[0]
                - aggregate.loc[aggregate.epsilon == 0.0, "baseline_posterior_rmse"].iloc[0]
            ),
            "median_test_time_ms_nonzero_contamination": float(runtime_nonzero.median()),
            "p95_test_time_ms_nonzero_contamination": float(runtime_nonzero.quantile(0.95)),
            "epsilon_0p5_rmse_reduction_pct": float(
                aggregate.loc[aggregate.epsilon == 0.5, "rmse_reduction_pct"].iloc[0]
            ),
        },
        "scope_boundary": (
            "Controlled conjugate-Gaussian CPU reproduction of the core RFF-MDS mechanism; "
            "not a rerun of every OUP, SIR, cryo-EM, NPE-PFN, or comparator experiment."
        ),
    }
    write_json(out / "experiment_summary.json", experiment_summary)
    return trials, aggregate, experiment_summary


def population_mmd_score(s: float, eps: float, y: float, gamma: float) -> float:
    clean = math.exp(-gamma * s * s / (1.0 + 4.0 * gamma)) / math.sqrt(1.0 + 4.0 * gamma)
    outlier = math.exp(-gamma * (s - y) ** 2 / (1.0 + 2.0 * gamma)) / math.sqrt(1.0 + 2.0 * gamma)
    return (1.0 - eps) * clean + eps * outlier


def population_mds(eps: float, y: float, gamma: float) -> float:
    # For the infinitesimal eps used here, the relevant/global basin is around 0.
    result = minimize_scalar(
        lambda s: -population_mmd_score(float(s), eps, y, gamma),
        bounds=(-4.0, 4.0),
        method="bounded",
        options={"xatol": 1e-13},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x)


def theorem_checks(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    gamma = 0.5
    y_dense = np.linspace(-50.0, 50.0, 20_001)
    a0 = 1.0 / math.sqrt(1.0 + 4.0 * gamma)
    ay0 = np.exp(-gamma * y_dense**2 / (1.0 + 2.0 * gamma)) / math.sqrt(1.0 + 2.0 * gamma)
    influence = y_dense * (1.0 + 4.0 * gamma) / (1.0 + 2.0 * gamma) * ay0 / a0
    max_abs_if = float(np.max(np.abs(influence)))
    y_at_max = float(y_dense[np.argmax(np.abs(influence))])

    # A finite-difference sanity check of Theorem 4.1's KL/epsilon quantity.
    y_eval = np.linspace(-25.0, 25.0, 501)
    post_var = 1.0 / (1.0 / PRIOR_VAR + N_OBS)
    post_factor = post_var * N_OBS
    influence_rows: list[dict] = []
    for eps in (1e-2, 1e-3, 1e-4):
        for y in y_eval:
            s = population_mds(eps, float(y), gamma)
            kl = (post_factor * s) ** 2 / (2.0 * post_var)
            influence_rows.append(
                {"epsilon": eps, "outlier_y": y, "mds_summary": s, "posterior_kl": kl, "kl_over_epsilon": kl / eps}
            )
    influence_df = pd.DataFrame(influence_rows)
    influence_df.to_csv(out / "robustness_influence_finite_difference.csv", index=False, float_format="%.12g")

    # Special-case posterior-consistency experiment for the exact predictive MMD.
    consistency_rows: list[dict] = []
    cons_rng = np.random.default_rng(314159)
    for n in (10, 30, 100, 300, 1000):
        pv = 1.0 / (1.0 / PRIOR_VAR + n)
        factor = pv * n
        predictive_var = 1.0 + pv
        for replicate in range(300):
            x = cons_rng.normal(0.0, 1.0, size=n)
            original_mean = factor * float(x.mean())
            # Optimize directly over the posterior/predictive location m.
            mds_mean = exact_conditional_mmd_summary(x, gamma, predictive_var)
            consistency_rows.append(
                {
                    "n": n,
                    "replicate": replicate,
                    "original_posterior_mean": original_mean,
                    "mds_posterior_mean": mds_mean,
                    "posterior_variance": pv,
                    "original_squared_radius": original_mean**2 + pv,
                    "mds_squared_radius": mds_mean**2 + pv,
                }
            )
    cons_trials = pd.DataFrame(consistency_rows)
    cons_trials.to_csv(out / "consistency_trials.csv", index=False, float_format="%.12g")
    cons_agg = (
        cons_trials.groupby("n", sort=True)
        .agg(
            trials=("replicate", "count"),
            original_mean_rmse=("original_posterior_mean", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            mds_mean_rmse=("mds_posterior_mean", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            posterior_std=("posterior_variance", lambda x: float(np.sqrt(np.mean(x)))),
            original_posterior_rms_radius=("original_squared_radius", lambda x: float(np.sqrt(np.mean(x)))),
            mds_posterior_rms_radius=("mds_squared_radius", lambda x: float(np.sqrt(np.mean(x)))),
        )
        .reset_index()
    )
    cons_agg.to_csv(out / "consistency_aggregate.csv", index=False, float_format="%.12g")
    slope = float(np.polyfit(np.log(cons_agg["n"]), np.log(cons_agg["mds_posterior_rms_radius"]), 1)[0])
    kl_max_by_eps = {
        f"{eps:.0e}": float(group["kl_over_epsilon"].max())
        for eps, group in influence_df.groupby("epsilon", sort=False)
    }
    theorem_summary = {
        "theorem_4_1_special_case": {
            "model": "P_x|s = Normal(s,1), Q = Normal(0,1), RBF gamma=0.5, point-mass Huber contamination",
            "analytic_sup_abs_summary_influence_over_y_grid": max_abs_if,
            "y_at_sup_abs_influence": y_at_max,
            "grid": [-50.0, 50.0, 20_001],
            "max_kl_over_epsilon_by_epsilon": kl_max_by_eps,
            "analytic_KL_derivative_at_epsilon_zero": 0.0,
            "interpretation": "Bounded summary influence and KL/epsilon -> 0 numerically in this regular Gaussian special case.",
        },
        "theorem_4_2_special_case": {
            "model": "Correctly specified Normal location model, exact Gaussian predictive MMD, RBF gamma=0.5",
            "n_values": cons_agg["n"].astype(int).tolist(),
            "replicates_per_n": 300,
            "initial_mds_posterior_rms_radius": float(cons_agg.iloc[0]["mds_posterior_rms_radius"]),
            "final_mds_posterior_rms_radius": float(cons_agg.iloc[-1]["mds_posterior_rms_radius"]),
            "log_log_slope": slope,
            "interpretation": "Posterior radius contracts toward zero as N increases; this is a numerical special case, not a proof of the general theorem.",
        },
        "source_theorems": {
            "theorem_4_1": "Infinitesimal KL stability under assumptions 1-10 in Appendix A.1.",
            "theorem_4_2": "Consistency transfer under a bounded characteristic kernel, continuity, and strong identifiability.",
            "important_boundary": "Theorem 4.1 is local/infinitesimal, not global posterior robustness; Theorem 4.2 does not cover decoder/NPE approximation error.",
        },
    }
    write_json(out / "theorem_checks_summary.json", theorem_summary)
    return influence_df, cons_agg, theorem_summary


def make_plots(out: Path, aggregate: pd.DataFrame, trials: pd.DataFrame, cons: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))

    ax = axes[0]
    ax.plot(aggregate.epsilon, aggregate.baseline_posterior_rmse, "o-", lw=2, label="NPE (observed mean)")
    ax.plot(aggregate.epsilon, aggregate.mds_rff_posterior_rmse, "o-", lw=2, label="NPE + official RFF-MDS")
    ax.plot(aggregate.epsilon, aggregate.mds_exact_posterior_rmse, "--", lw=1.5, label="Exact MMD reference")
    ax.set(xlabel="Contamination fraction", ylabel="Posterior-mean RMSE", title="Robustness across 50 seeds")
    ax.legend(fontsize=8)

    ax = axes[1]
    nonzero = trials.query("epsilon > 0")
    data = [nonzero.loc[nonzero.epsilon == eps, "adapt_ms"] for eps in EPSILONS[1:]]
    ax.boxplot(data, tick_labels=[f"{e:.1f}" for e in EPSILONS[1:]], showfliers=False)
    ax.set(xlabel="Contamination fraction", ylabel="CPU test-time adaptation (ms)", title="Lightweight deterministic adaptation")

    ax = axes[2]
    ax.loglog(cons.n, cons.original_posterior_rms_radius, "o-", lw=2, label="Original summary")
    ax.loglog(cons.n, cons.mds_posterior_rms_radius, "o-", lw=2, label="Exact MDS")
    ax.set(xlabel="N", ylabel="Posterior RMS radius", title="Consistency special case")
    ax.legend(fontsize=8)

    fig.suptitle("Minimum Distance Summaries - CPU-only reproduction", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "robustness_cost_consistency.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def environment_payload() -> dict:
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "device": "cpu",
        "gpu_used": False,
        "network_used_during_reproduction": False,
        "paid_services_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "full")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    audit = source_audit(out)
    trials, aggregate, experiment = run_official_rff_experiment(out)
    _, consistency, theorem = theorem_checks(out)
    make_plots(out, aggregate, trials, consistency)
    env = environment_payload()
    env["wall_seconds_total"] = time.perf_counter() - started
    write_json(out / "environment.json", env)

    summary = {
        "paper_id": PAPER_ID,
        "title": TITLE,
        "arxiv_id": "2602.09161",
        "claims": {
            "claim_1_plugin_mds_decoupled_from_pretrained_npe": {
                "verdict": "verified",
                "evidence": "Source Algorithm 1 plus unchanged analytic NPE queried before/after official RFF-MDS across 300 CPU trials.",
            },
            "claim_2_rff_lightweight_model_free_adaptation": {
                "verdict": "verified_with_packaging_caveats",
                "evidence": {
                    "paper_scale_rff_dimension": 512,
                    "official_code_patterns_all_present": all(audit["official_code_patterns"].values()),
                    "median_test_time_ms": experiment["headline"]["median_test_time_ms_nonzero_contamination"],
                    "p95_test_time_ms": experiment["headline"]["p95_test_time_ms_nonzero_contamination"],
                },
            },
            "claim_3_robustness_gain_minimal_overhead": {
                "verdict": "verified_in_controlled_gaussian_reproduction",
                "evidence": experiment["headline"],
            },
            "claim_4_theoretical_guarantees": {
                "verdict": "verified_with_stated_scope",
                "evidence": theorem,
            },
        },
        "cpu_only": True,
        "wall_seconds_total": env["wall_seconds_total"],
        "raw_files": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

````


````output
Calibrated tau=0.003727 (alpha=0.05, n_calib=100)
  Loss stats: mean=0.002068, std=0.006231
{
  "arxiv_id": "2602.09161",
  "claims": {
    "claim_1_plugin_mds_decoupled_from_pretrained_npe": {
      "evidence": "Source Algorithm 1 plus unchanged analytic NPE queried before/after official RFF-MDS across 300 CPU trials.",
      "verdict": "verified"
    },
    "claim_2_rff_lightweight_model_free_adaptation": {
      "evidence": {
        "median_test_time_ms": 7.11321400012821,
        "official_code_patterns_all_present": true,
        "p95_test_time_ms": 14.595631230622523,
        "paper_scale_rff_dimension": 512
      },
      "verdict": "verified_with_packaging_caveats"
    },
    "claim_3_robustness_gain_minimal_overhead": {
      "evidence": {
        "clean_gate_pass_rate": 1.0,
        "clean_rmse_delta": 8.90399698416644e-11,
        "epsilon_0p5_rmse_reduction_pct": 4.787768185186102,
        "mean_rmse_reduction_pct_eps_0p1_to_0p4": 91.40844079367253,
        "median_test_time_ms_nonzero_contamination": 7.11321400012821,
        "minimum_rmse_reduction_pct_eps_0p1_to_0p4": 86.0722712342806,
        "minimum_win_rate_eps_0p1_to_0p4": 1.0,
        "p95_test_time_ms_nonzero_contamination": 14.595631230622523
      },
      "verdict": "verified_in_controlled_gaussian_reproduction"
    },
    "claim_4_theoretical_guarantees": {
      "evidence": {
        "source_theorems": {
          "important_boundary": "Theorem 4.1 is local/infinitesimal, not global posterior robustness; Theorem 4.2 does not cover decoder/NPE approximation error.",
          "theorem_4_1": "Infinitesimal KL stability under assumptions 1-10 in Appendix A.1.",
          "theorem_4_2": "Consistency transfer under a bounded characteristic kernel, continuity, and strong identifiability."
        },
        "theorem_4_1_special_case": {
          "analytic_KL_derivative_at_epsilon_zero": 0.0,
          "analytic_sup_abs_summary_influence_over_y_grid": 1.575812391238556,
          "grid": [
            -50.0,
            50.0,
            20001
          ],
          "interpretation": "Bounded summary influence and KL/epsilon -> 0 numerically in this regular Gaussian special case.",
          "max_kl_over_epsilon_by_epsilon": {
            "1e-02": 1.2625929291602669,
            "1e-03": 0.12406538826146586,
            "1e-04": 0.01238738159765737
          },
          "model": "P_x|s = Normal(s,1), Q = Normal(0,1), RBF gamma=0.5, point-mass Huber contamination",
          "y_at_sup_abs_influence": -1.4149999999999991
        },
        "theorem_4_2_special_case": {
          "final_mds_posterior_rms_radius": 0.0455683260916302,
          "initial_mds_posterior_rms_radius": 0.4659525769912868,
          "interpretation": "Posterior radius contracts toward zero as N increases; this is a numerical special case, not a proof of the general theorem.",
          "log_log_slope": -0.5074428753415307,
          "model": "Correctly specified Normal location model, exact Gaussian predictive MMD, RBF gamma=0.5",
          "n_values": [
            10,
            30,
            100,
            300,
            1000
          ],
          "replicates_per_n": 300
        }
      },
      "verdict": "verified_with_stated_scope"
    }
  },
  "cpu_only": true,
  "paper_id": "lq8fNVME8v",
  "raw_files": [
    "consistency_aggregate.csv",
    "consistency_trials.csv",
    "environment.json",
    "experiment_summary.json",
    "gaussian_aggregate.csv",
    "gaussian_neural_posterior.pt",
    "gaussian_neural_training.csv",
    "gaussian_trials.csv",
    "neural_gaussian_aggregate.csv",
    "neural_gaussian_trials.csv",
    "neural_oup_aggregate.csv",
    "neural_oup_trials.csv",
    "neural_upgrade_summary.json",
    "oup_neural_posterior.pt",
    "oup_neural_training.csv",
    "proof_certificate.json",
    "robustness_cost_consistency.png",
    "robustness_influence_finite_difference.csv",
    "source_and_code_audit.json",
    "summary.json",
    "theorem_checks_summary.json"
  ],
  "title": "Minimum Distance Summaries for Robust Neural Posterior Estimation",
  "wall_seconds_total": 15.814923701109365
}

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_f3aeb70bccaf", "created_at": "2026-07-16T15:52:37+00:00", "title": "Artifact: consistency_trials.csv", "path": "outputs/full/consistency_trials.csv", "size": 133042, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/full/consistency_trials.csv` · dataset · 0.1 MB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/full/consistency_trials.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_39b023688656", "created_at": "2026-07-16T15:52:37+00:00", "title": "Artifact: gaussian_trials.csv", "path": "outputs/full/gaussian_trials.csv", "size": 104073, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/full/gaussian_trials.csv` · dataset · 0.1 MB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/full/gaussian_trials.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_8651da1edebe", "created_at": "2026-07-16T15:52:37+00:00", "title": "Artifact: robustness_influence_finite_difference.csv", "path": "outputs/full/robustness_influence_finite_difference.csv", "size": 97833, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/full/robustness_influence_finite_difference.csv` · dataset · 97.8 kB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/full/robustness_influence_finite_difference.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_39632d7aa3f2", "created_at": "2026-07-16T15:52:37+00:00", "title": "Artifact: gaussian_aggregate.csv", "path": "outputs/full/gaussian_aggregate.csv", "size": 1633, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/full/gaussian_aggregate.csv` · dataset · 1.6 kB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/full/gaussian_aggregate.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_46239b2d28a2", "created_at": "2026-07-16T15:52:37+00:00", "title": "Artifact: consistency_aggregate.csv", "path": "outputs/full/consistency_aggregate.csv", "size": 530, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/full/consistency_aggregate.csv` · dataset · 530 B

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/full/consistency_aggregate.csv


---
<!-- trackio-cell
{"type": "code", "id": "cell_1f46d1850a1a", "created_at": "2026-07-16T15:52:38+00:00", "title": "Dimension-general proof audit", "command": [".venv/bin/python", "reproduction/audit_dimension_general_proof.py"], "exit_code": 0, "duration_s": 0.043}
-->
````bash
$ .venv/bin/python reproduction/audit_dimension_general_proof.py
````

exit 0 · 0.0s


````python title=audit_dimension_general_proof.py
#!/usr/bin/env python3
"""Fail-closed dependency audit of the paper's dimension-general proofs.

This is not a mechanized proof assistant and does not re-prove imported
theorems. It checks that each implication used by Theorems 4.1 and 4.2 is
present in the pinned TeX, records the exact assumptions/equations that license
it, verifies the dependency DAG, and flags notational/interpretive gaps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "source" / "arxiv" / "arxiv_main.tex"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "full" / "proof_certificate.json")
    args = parser.parse_args()
    lines = TEX.read_text(encoding="utf-8").splitlines()

    def locate(needle: str, start: int = 1) -> int:
        hits = [i + 1 for i, line in enumerate(lines) if i + 1 >= start and needle in line]
        if not hits:
            raise AssertionError(f"Pinned TeX anchor missing after line {start}: {needle}")
        return hits[0]

    anchors = {
        "population_mds": locate("population MMD based minimum-distance summary objective"),
        "contamination_path": locate("define the contaminated distribution"),
        "robustness_statement": locate("\\begin{theorem} \\label{thm:robustness}"),
        "robustness_assumptions": locate("Assumptions for Theorem \\ref{thm:robustness}"),
        "bounded_kernel": locate("\\item \\label{itm:infl1}"),
        "derivative_bound": locate("\\item \\label{itm:deriv_bound}"),
        "nonsingular_hessian": locate("\\item \\label{itm:non_sing}"),
        "influence_definition": locate("Define the influence function"),
        "influence_formula": locate("= M(\\mathbf{s}^*(\\mathbb{Q}))^{-1}"),
        "dimension_sum_bound": locate("4 \\sup_{\\mathbf{z}, \\mathbf{z}'}"),
        "summary_likelihood_mvt": locate("By the mean value theorem"),
        "rob2": locate("\\begin{equation} \\label{eq:rob2}"),
        "rob3_kl_stability": locate("\\begin{equation} \\label{eq:rob3}"),
        "rob4_bounded_influence": locate("\\begin{equation} \\label{eq:rob4}"),
        "robustness_quotient_conclusion": locate("Using \\eqref{eq:rob4} gives the required result"),
        "consistency_statement": locate("\\begin{theorem} \\label{thm:consistency}"),
        "consistency_assumptions": locate("Assumptions for Theorem \\ref{thm:consistency}"),
        "kernel_metrizes": locate("ensures that MMD metrizes weak convergence"),
        "exact_conditionals_boundary": locate("This result does not consider approximation error"),
        "mds_argmin": locate("\\mathbf{s}_N^* = \\argmin"),
        "two_weak_limits": locate("We will need two weak convergence requirements"),
        "consistency_part_1": locate("\\paragraph{Part 1}", 700),
        "consistency_part_2": locate("\\paragraph{Part 2}", 700),
        "consistency_part_3": locate("\\paragraph{Part 3}", 700),
        "argmin_contraction": locate("A_N \\leq B_N"),
        "consistency_part_4": locate("\\paragraph{Part 4}", 700),
    }

    robust_segment = "\n".join(lines[anchors["robustness_assumptions"] - 1 : anchors["influence_definition"] - 2])
    consistency_segment = "\n".join(lines[anchors["consistency_assumptions"] - 1 : anchors["kernel_metrizes"] - 1])
    robust_assumption_count = robust_segment.count("\\item")
    consistency_assumption_count = consistency_segment.count("\\item")
    assert robust_assumption_count == 10, robust_assumption_count
    assert consistency_assumption_count == 4, consistency_assumption_count
    assert "d_\\mathbf{s}" in robust_segment
    assert "h: \\mathbb{R}^{d_x}" in "\n".join(lines[740:760])

    steps = [
        {
            "id": "R0",
            "name": "Define the population perturbation path",
            "lines": [anchors["population_mds"], anchors["contamination_path"]],
            "depends_on": [],
            "premises": "An MMD minimizer exists; Q_eps,y=(1-eps)Q+eps delta_y.",
            "audited_inference": "The theorem studies the one-sided path through the population MDS functional, not finite-sample RFF optimization.",
        },
        {
            "id": "R1",
            "name": "Bound the dimension-general MDS influence",
            "lines": [anchors["bounded_kernel"], anchors["dimension_sum_bound"]],
            "depends_on": ["R0"],
            "premises": "Assumptions 6-10: bounded kernel; density p_s; C2 summary dependence; integrable coordinate derivatives; nonsingular local Hessian M.",
            "audited_inference": "IF(y;Q)=M^{-1} grad_s xi. Dominated differentiation and the triangle inequality give ||grad_s xi|| <= 4||k||_infty sum_{i=1}^{d_s} int|partial_i p_s|, a finite bound independent of y.",
            "dimension_general": "The bound explicitly sums i=1,...,d_s and never fixes d_s or d_x.",
        },
        {
            "id": "R2",
            "name": "Transfer summary perturbation to likelihood perturbation",
            "lines": [anchors["summary_likelihood_mvt"], anchors["rob2"]],
            "depends_on": ["R1"],
            "premises": "Assumptions 3-5: Phi_s differentiable in vector s, locally integrable dominating gradient, convex S.",
            "audited_inference": "The vector mean-value path remains in S; integrating the gradient bound yields ||Phi_s-Phi_t||_L1(mu) <= k1(s)||s-t|| locally.",
        },
        {
            "id": "R3",
            "name": "Transfer likelihood perturbation to posterior KL",
            "lines": [anchors["rob3_kl_stability"], anchors["robustness_quotient_conclusion"]],
            "depends_on": ["R2"],
            "premises": "Assumptions 1-2 put the posterior in normalized exp(-Phi_s) form and normalize ess inf Phi_s=0; imported Sprungk local KL stability applies when the L1 perturbation is <=1.",
            "audited_inference": "For sufficiently small eps, KL/eps <= k1*k2*||s*(Q_eps,y)-s*(Q)||/eps; R1 makes the right side uniformly finite in y.",
        },
        {
            "id": "C1",
            "name": "Original-posterior consistency implies predictive consistency",
            "lines": [anchors["two_weak_limits"], anchors["consistency_part_1"]],
            "depends_on": [],
            "premises": "P(theta|s_N) => delta_theta0; empirical law => P_x|theta0 almost surely; G_theta(u) continuous in theta.",
            "audited_inference": "For every bounded continuous h on R^{d_x}, continuous mapping plus dominated convergence gives E_{predictive(s_N)}h -> E_{P_theta0}h, which is weak predictive convergence.",
        },
        {
            "id": "C2",
            "name": "Convert weak convergence to vanishing MMD",
            "lines": [anchors["kernel_metrizes"], anchors["consistency_part_2"]],
            "depends_on": ["C1"],
            "premises": "The kernel is bounded, continuous and integrally strictly positive definite, so the imported metrization result applies.",
            "audited_inference": "Both predictive(s_N) and the empirical measure approach P_theta0; the MMD triangle inequality gives B_N=MMD(predictive(s_N),P_hat_N)->0.",
        },
        {
            "id": "C3",
            "name": "Use minimization to transfer convergence to MDS",
            "lines": [anchors["mds_argmin"], anchors["argmin_contraction"]],
            "depends_on": ["C2"],
            "premises": "The MDS argmin exists.",
            "audited_inference": "A_N=MMD(predictive(s*_N),P_hat_N) <= B_N by optimality; squeeze gives A_N->0, then a second triangle inequality gives predictive(s*_N)=>P_theta0.",
        },
        {
            "id": "C4",
            "name": "Apply mixture identifiability",
            "lines": [anchors["consistency_part_3"], anchors["consistency_part_4"]],
            "depends_on": ["C3"],
            "premises": "Assumption 4: convergence of predictive mixtures to P_x|theta0 forces their mixing measures to delta_theta0.",
            "audited_inference": "Taking T_N=P(theta|s*_N), predictive convergence from C3 yields P(theta|s*_N)=>delta_theta0.",
        },
    ]

    ids = {s["id"] for s in steps}
    assert len(ids) == len(steps)
    resolved: set[str] = set()
    for step in steps:
        assert set(step["depends_on"]) <= resolved, (step["id"], step["depends_on"], resolved)
        resolved.add(step["id"])

    certificate = {
        "status": "PASS",
        "audit_kind": "dimension-general proof dependency and inequality audit; not a mechanized proof and not an empirical substitute",
        "source": {"path": "source/arxiv/arxiv_main.tex", "sha256": sha256(TEX), "line_count": len(lines)},
        "scope": {
            "robustness": "Population MDS, local/infinitesimal point-mass Huber contamination, arbitrary finite d_s and d_x under ten stated assumptions.",
            "consistency": "Exact regular conditionals, arbitrary finite d_x, bounded characteristic-kernel metrization, continuous simulator, argmin existence and strong mixture identifiability.",
        },
        "anchor_lines": anchors,
        "assumption_counts": {"robustness": robust_assumption_count, "consistency": consistency_assumption_count},
        "dependency_steps": steps,
        "independent_audit_findings": [
            {
                "severity": "notation_only",
                "location": 601,
                "finding": "The displayed gradient bound writes sup k; the triangle-inequality derivation requires sup |k|. Assumption 6 says the kernel is bounded, so replacing this with ||k||_infinity is licensed and adds no assumption.",
            },
            {
                "severity": "typographical",
                "location": 656,
                "finding": "One intermediate TeX expression drops the star from s*(Q_eps,y); the immediately surrounding equations use s* and the substitution is unambiguous.",
            },
            {
                "severity": "interpretive",
                "location": [341, 565, 680],
                "finding": "The theorem writes a derivative while the appendix bounds a one-sided KL difference quotient. Identifying them uses KL(0)=0 and existence of the influence-function limit invoked at lines 562-580; the certificate does not claim a stronger two-sided/global derivative result.",
            },
            {
                "severity": "scope",
                "location": 720,
                "finding": "The consistency proof explicitly excludes NPE/decoder approximation error; it certifies exact regular conditionals only.",
            },
        ],
        "logical_result": {
            "robustness": "PASS: every link R0->R1->R2->R3 is present and assumption-licensed after the harmless sup|k| notation repair.",
            "consistency": "PASS: every link C1->C2->C3->C4 is present; the final step depends essentially on the stated strong identifiability assumption.",
            "overclaim_guard": "This certificate checks the pinned paper's proof dependency chain. It does not validate imported theorems beyond their stated use, does not cover learned approximation error, and does not imply global robustness.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(certificate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

````


````output
{
  "anchor_lines": {
    "argmin_contraction": 791,
    "bounded_kernel": 537,
    "consistency_assumptions": 686,
    "consistency_part_1": 744,
    "consistency_part_2": 761,
    "consistency_part_3": 775,
    "consistency_part_4": 794,
    "consistency_statement": 368,
    "contamination_path": 334,
    "derivative_bound": 540,
    "dimension_sum_bound": 601,
    "exact_conditionals_boundary": 720,
    "influence_definition": 562,
    "influence_formula": 580,
    "kernel_metrizes": 706,
    "mds_argmin": 726,
    "nonsingular_hessian": 542,
    "population_mds": 326,
    "rob2": 625,
    "rob3_kl_stability": 635,
    "rob4_bounded_influence": 646,
    "robustness_assumptions": 497,
    "robustness_quotient_conclusion": 678,
    "robustness_statement": 337,
    "summary_likelihood_mvt": 613,
    "two_weak_limits": 732
  },
  "assumption_counts": {
    "consistency": 4,
    "robustness": 10
  },
  "audit_kind": "dimension-general proof dependency and inequality audit; not a mechanized proof and not an empirical substitute",
  "dependency_steps": [
    {
      "audited_inference": "The theorem studies the one-sided path through the population MDS functional, not finite-sample RFF optimization.",
      "depends_on": [],
      "id": "R0",
      "lines": [
        326,
        334
      ],
      "name": "Define the population perturbation path",
      "premises": "An MMD minimizer exists; Q_eps,y=(1-eps)Q+eps delta_y."
    },
    {
      "audited_inference": "IF(y;Q)=M^{-1} grad_s xi. Dominated differentiation and the triangle inequality give ||grad_s xi|| <= 4||k||_infty sum_{i=1}^{d_s} int|partial_i p_s|, a finite bound independent of y.",
      "depends_on": [
        "R0"
      ],
      "dimension_general": "The bound explicitly sums i=1,...,d_s and never fixes d_s or d_x.",
      "id": "R1",
      "lines": [
        537,
        601
      ],
      "name": "Bound the dimension-general MDS influence",
      "premises": "Assumptions 6-10: bounded kernel; density p_s; C2 summary dependence; integrable coordinate derivatives; nonsingular local Hessian M."
    },
    {
      "audited_inference": "The vector mean-value path remains in S; integrating the gradient bound yields ||Phi_s-Phi_t||_L1(mu) <= k1(s)||s-t|| locally.",
      "depends_on": [
        "R1"
      ],
      "id": "R2",
      "lines": [
        613,
        625
      ],
      "name": "Transfer summary perturbation to likelihood perturbation",
      "premises": "Assumptions 3-5: Phi_s differentiable in vector s, locally integrable dominating gradient, convex S."
    },
    {
      "audited_inference": "For sufficiently small eps, KL/eps <= k1*k2*||s*(Q_eps,y)-s*(Q)||/eps; R1 makes the right side uniformly finite in y.",
      "depends_on": [
        "R2"
      ],
      "id": "R3",
      "lines": [
        635,
        678
      ],
      "name": "Transfer likelihood perturbation to posterior KL",
      "premises": "Assumptions 1-2 put the posterior in normalized exp(-Phi_s) form and normalize ess inf Phi_s=0; imported Sprungk local KL stability applies when the L1 perturbation is <=1."
    },
    {
      "audited_inference": "For every bounded continuous h on R^{d_x}, continuous mapping plus dominated convergence gives E_{predictive(s_N)}h -> E_{P_theta0}h, which is weak predictive convergence.",
      "depends_on": [],
      "id": "C1",
      "lines": [
        732,
        744
      ],
      "name": "Original-posterior consistency implies predictive consistency",
      "premises": "P(theta|s_N) => delta_theta0; empirical law => P_x|theta0 almost surely; G_theta(u) continuous in theta."
    },
    {
      "audited_inference": "Both predictive(s_N) and the empirical measure approach P_theta0; the MMD triangle inequality gives B_N=MMD(predictive(s_N),P_hat_N)->0.",
      "depends_on": [
        "C1"
      ],
      "id": "C2",
      "lines": [
        706,
        761
      ],
      "name": "Convert weak convergence to vanishing MMD",
      "premises": "The kernel is bounded, continuous and integrally strictly positive definite, so the imported metrization result applies."
    },
    {
      "audited_inference": "A_N=MMD(predictive(s*_N),P_hat_N) <= B_N by optimality; squeeze gives A_N->0, then a second triangle inequality gives predictive(s*_N)=>P_theta0.",
      "depends_on": [
        "C2"
      ],
      "id": "C3",
      "lines": [
        726,
        791
      ],
      "name": "Use minimization to transfer convergence to MDS",
      "premises": "The MDS argmin exists."
    },
    {
      "audited_inference": "Taking T_N=P(theta|s*_N), predictive convergence from C3 yields P(theta|s*_N)=>delta_theta0.",
      "depends_on": [
        "C3"
      ],
      "id": "C4",
      "lines": [
        775,
        794
      ],
      "name": "Apply mixture identifiability",
      "premises": "Assumption 4: convergence of predictive mixtures to P_x|theta0 forces their mixing measures to delta_theta0."
    }
  ],
  "independent_audit_findings": [
    {
      "finding": "The displayed gradient bound writes sup k; the triangle-inequality derivation requires sup |k|. Assumption 6 says the kernel is bounded, so replacing this with ||k||_infinity is licensed and adds no assumption.",
      "location": 601,
      "severity": "notation_only"
    },
    {
      "finding": "One intermediate TeX expression drops the star from s*(Q_eps,y); the immediately surrounding equations use s* and the substitution is unambiguous.",
      "location": 656,
      "severity": "typographical"
    },
    {
      "finding": "The theorem writes a derivative while the appendix bounds a one-sided KL difference quotient. Identifying them uses KL(0)=0 and existence of the influence-function limit invoked at lines 562-580; the certificate does not claim a stronger two-sided/global derivative result.",
      "location": [
        341,
        565,
        680
      ],
      "severity": "interpretive"
    },
    {
      "finding": "The consistency proof explicitly excludes NPE/decoder approximation error; it certifies exact regular conditionals only.",
      "location": 720,
      "severity": "scope"
    }
  ],
  "logical_result": {
    "consistency": "PASS: every link C1->C2->C3->C4 is present; the final step depends essentially on the stated strong identifiability assumption.",
    "overclaim_guard": "This certificate checks the pinned paper's proof dependency chain. It does not validate imported theorems beyond their stated use, does not cover learned approximation error, and does not imply global robustness.",
    "robustness": "PASS: every link R0->R1->R2->R3 is present and assumption-licensed after the harmless sup|k| notation repair."
  },
  "scope": {
    "consistency": "Exact regular conditionals, arbitrary finite d_x, bounded characteristic-kernel metrization, continuous simulator, argmin existence and strong mixture identifiability.",
    "robustness": "Population MDS, local/infinitesimal point-mass Huber contamination, arbitrary finite d_s and d_x under ten stated assumptions."
  },
  "source": {
    "line_count": 1197,
    "path": "source/arxiv/arxiv_main.tex",
    "sha256": "ff81fd973e3bcba86fb23e9a0c102ec88e240f62361315c7875de54e29ea4fd2"
  },
  "status": "PASS"
}

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_67c3b08b72fa", "created_at": "2026-07-16T15:53:33+00:00", "title": "Frozen neural NPE protocol", "command": [".venv/bin/python", "reproduction/neural_npe_upgrade.py"], "exit_code": 0, "duration_s": 54.832}
-->
````bash
$ .venv/bin/python reproduction/neural_npe_upgrade.py
````

exit 0 · 54.8s


````python title=neural_npe_upgrade.py
#!/usr/bin/env python3
"""Actual neural-posterior and second-task CPU upgrade for the MDS logbook.

The original reproduction used an exact conjugate posterior so that the MDS
adapter could be isolated.  This upgrade addresses that boundary directly:
two conditional Gaussian neural posterior estimators are trained, frozen, and
queried with either the ordinary or MDS-adapted summary.  The second mechanism
is the authors' Ornstein-Uhlenbeck-process (OUP) simulator and moment summary.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import torch
from torch import nn

# Tiny CPU L-BFGS queries are latency-bound; a large OpenMP pool makes each
# objective evaluation substantially slower on many-core hosts.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "source" / "official-repo"
OFFICIAL_RFF = OFFICIAL / "src" / "tt_sbi" / "tta" / "rff.py"
OFFICIAL_OUP = OFFICIAL / "src" / "tt_sbi" / "tasks" / "oup.py"
PROTOCOL = ROOT / "reproduction" / "NEURAL_UPGRADE_PROTOCOL.json"
SEED = 20260716


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_rff_module():
    spec = importlib.util.spec_from_file_location("official_mds_rff_upgrade", OFFICIAL_RFF)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned official RFF module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        array = value.detach().cpu().contiguous().numpy()
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


class ConditionalDiagonalGaussian(nn.Module):
    """Small conditional density network: q_phi(theta | summary)."""

    def __init__(self, input_dim: int, theta_dim: int, hidden: int):
        super().__init__()
        self.theta_dim = theta_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2 * theta_dim),
        )

    def forward(self, summary_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.net(summary_z)
        mean, log_std = output[..., : self.theta_dim], output[..., self.theta_dim :]
        return mean, log_std.clamp(-5.0, 2.0)


@dataclass
class FitResult:
    best_epoch: int
    epochs_ran: int
    best_val_nll: float
    train_seconds: float
    parameter_count: int
    model_sha256: str
    summary_mean: list[float]
    summary_std: list[float]
    theta_mean: list[float]
    theta_std: list[float]


def fit_neural_posterior(
    summaries: torch.Tensor,
    thetas: torch.Tensor,
    *,
    hidden: int,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    seed: int,
    checkpoint: Path,
    curve_path: Path,
) -> tuple[ConditionalDiagonalGaussian, dict[str, torch.Tensor], FitResult]:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    summaries = summaries.float().cpu()
    thetas = thetas.float().cpu()
    n = len(summaries)
    split_gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=split_gen)
    n_val = max(100, int(0.15 * n))
    val_idx, train_idx = order[:n_val], order[n_val:]
    s_mean = summaries[train_idx].mean(0)
    s_std = summaries[train_idx].std(0).clamp_min(1e-6)
    t_mean = thetas[train_idx].mean(0)
    t_std = thetas[train_idx].std(0).clamp_min(1e-6)
    s_z = (summaries - s_mean) / s_std
    t_z = (thetas - t_mean) / t_std

    model = ConditionalDiagonalGaussian(summaries.shape[1], thetas.shape[1], hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    curve: list[dict] = []
    started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        epoch_gen = torch.Generator().manual_seed(seed + 1_000 + epoch)
        shuffled = train_idx[torch.randperm(len(train_idx), generator=epoch_gen)]
        train_sum = 0.0
        train_items = 0
        for start in range(0, len(shuffled), batch_size):
            idx = shuffled[start : start + batch_size]
            mean, log_std = model(s_z[idx])
            nll = (0.5 * ((t_z[idx] - mean) / log_std.exp()) ** 2 + log_std).sum(1).mean()
            optimizer.zero_grad()
            nll.backward()
            optimizer.step()
            train_sum += float(nll.detach()) * len(idx)
            train_items += len(idx)

        model.eval()
        with torch.no_grad():
            mean, log_std = model(s_z[val_idx])
            val = (0.5 * ((t_z[val_idx] - mean) / log_std.exp()) ** 2 + log_std).sum(1).mean()
        train_nll = train_sum / train_items
        val_nll = float(val)
        curve.append({"epoch": epoch + 1, "train_nll": train_nll, "val_nll": val_nll})
        if val_nll < best_val - 1e-6:
            best_val = val_nll
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise AssertionError("neural posterior never produced a finite validation loss")
    model.load_state_dict(best_state)
    model.eval()
    stats = {"s_mean": s_mean, "s_std": s_std, "t_mean": t_mean, "t_std": t_std}
    torch.save(
        {
            "architecture": "conditional diagonal Gaussian neural posterior",
            "state_dict": model.state_dict(),
            "normalization": stats,
            "seed": seed,
        },
        checkpoint,
    )
    pd.DataFrame(curve).to_csv(curve_path, index=False, float_format="%.12g")
    result = FitResult(
        best_epoch=best_epoch,
        epochs_ran=len(curve),
        best_val_nll=best_val,
        train_seconds=time.perf_counter() - started,
        parameter_count=sum(p.numel() for p in model.parameters()),
        model_sha256=tensor_state_hash(model),
        summary_mean=s_mean.tolist(),
        summary_std=s_std.tolist(),
        theta_mean=t_mean.tolist(),
        theta_std=t_std.tolist(),
    )
    return model, stats, result


def posterior_mean(model: nn.Module, stats: dict[str, torch.Tensor], summary: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        summary_z = (summary.float() - stats["s_mean"]) / stats["s_std"]
        mean_z, _ = model(summary_z)
        return mean_z * stats["t_std"] + stats["t_mean"]


def gaussian_training_data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    theta = 2.0 * torch.randn(n, 1, generator=gen)
    summary = theta + 0.1 * torch.randn(n, 1, generator=gen)
    return summary, theta


def evaluate_gaussian_neural(
    model: nn.Module, stats: dict[str, torch.Tensor], out: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(ROOT / "outputs" / "full" / "gaussian_trials.csv")
    base_s = torch.tensor(source[["s_observed"]].to_numpy(), dtype=torch.float32)
    mds_s = torch.tensor(source[["s_mds_rff"]].to_numpy(), dtype=torch.float32)
    base_mean = posterior_mean(model, stats, base_s).squeeze(1).numpy()
    mds_mean = posterior_mean(model, stats, mds_s).squeeze(1).numpy()
    rows = source[["epsilon", "seed", "s_observed", "s_mds_rff", "adapt_ms"]].copy()
    rows["theta_true"] = 0.0
    rows["npe_mean_baseline"] = base_mean
    rows["npe_mean_mds"] = mds_mean
    rows["abs_error_baseline"] = np.abs(base_mean)
    rows["abs_error_mds"] = np.abs(mds_mean)
    rows.to_csv(out / "neural_gaussian_trials.csv", index=False, float_format="%.12g")

    aggregate = []
    for eps, group in rows.groupby("epsilon", sort=True):
        base = group["npe_mean_baseline"].to_numpy()
        mds = group["npe_mean_mds"].to_numpy()
        delta = np.abs(base) - np.abs(mds)
        base_rmse = float(np.sqrt(np.mean(base**2)))
        mds_rmse = float(np.sqrt(np.mean(mds**2)))
        p = 1.0 if np.allclose(delta, 0) else float(ttest_rel(np.abs(base), np.abs(mds)).pvalue)
        aggregate.append(
            {
                "epsilon": float(eps),
                "trials": len(group),
                "neural_npe_baseline_rmse": base_rmse,
                "neural_npe_mds_rmse": mds_rmse,
                "rmse_reduction_pct": 100 * (base_rmse - mds_rmse) / max(base_rmse, 1e-12),
                "paired_win_rate": float(np.mean(np.abs(mds) < np.abs(base))),
                "paired_ttest_pvalue": p,
                "median_adapt_ms": float(group["adapt_ms"].median()),
            }
        )
    agg = pd.DataFrame(aggregate)
    agg.to_csv(out / "neural_gaussian_aggregate.csv", index=False, float_format="%.12g")
    return rows, agg


@dataclass(frozen=True)
class OUPConfig:
    n_timesteps: int = 25
    n_trajectories: int = 100
    variance_scale: float = 0.1
    horizon: float = 5.0
    y0: float = 10.0


def sample_oup_prior(n: int, gen: torch.Generator) -> torch.Tensor:
    theta1 = 2.0 * torch.rand(n, generator=gen)
    theta2 = -2.0 + 4.0 * torch.rand(n, generator=gen)
    return torch.stack([theta1, theta2], dim=1)


def simulate_oup(
    theta: torch.Tensor, cfg: OUPConfig, gen: torch.Generator, *, variance_scale: float | None = None
) -> torch.Tensor:
    n = len(theta)
    y = torch.zeros(n, cfg.n_trajectories, cfg.n_timesteps)
    y[:, :, 0] = cfg.y0
    current = y[:, :, 0]
    dt = cfg.horizon / (cfg.n_timesteps + 1)
    scale = cfg.variance_scale if variance_scale is None else variance_scale
    theta1 = theta[:, 0].unsqueeze(1)
    theta2 = torch.exp(theta[:, 1]).unsqueeze(1)
    for step in range(cfg.n_timesteps - 1):
        noise = torch.randn(n, cfg.n_trajectories, generator=gen) * scale
        current = current + theta1 * (theta2 - current) * dt + 0.5 * math.sqrt(dt) * noise
        y[:, :, step + 1] = current
    return y


def oup_summary(y: torch.Tensor) -> torch.Tensor:
    k = max(5, y.shape[-1] // 3)
    window = y[:, :, -k:]
    mu = window.mean(dim=(1, 2))
    var = window.var(dim=(1, 2), correction=0)
    x = window[:, :, :-1].reshape(len(y), -1)
    z = window[:, :, 1:].reshape(len(y), -1)
    x0, z0 = x - x.mean(1, keepdim=True), z - z.mean(1, keepdim=True)
    corr = (x0 * z0).mean(1) / torch.sqrt(
        (x0.square().mean(1) * z0.square().mean(1)).clamp_min(1e-16)
    )
    return torch.stack([mu, var, corr], dim=1)


def generate_oup_training_summaries(
    n: int, cfg: OUPConfig, seed: int, batch: int = 250
) -> tuple[torch.Tensor, torch.Tensor]:
    prior_gen = torch.Generator().manual_seed(seed)
    sim_gen = torch.Generator().manual_seed(seed + 1)
    theta_parts, summary_parts = [], []
    for start in range(0, n, batch):
        size = min(batch, n - start)
        theta = sample_oup_prior(size, prior_gen)
        summary_parts.append(oup_summary(simulate_oup(theta, cfg, sim_gen)))
        theta_parts.append(theta)
    return torch.cat(summary_parts), torch.cat(theta_parts)


def fit_oup_rff(cfg: OUPConfig, n: int, rff_dim: int, epochs: int):
    module = load_rff_module()
    prior_gen = torch.Generator().manual_seed(SEED + 40)
    sim_gen = torch.Generator().manual_seed(SEED + 41)
    theta = sample_oup_prior(n, prior_gen)
    x = simulate_oup(theta, cfg, sim_gen)
    summary = oup_summary(x)
    adapter = module.RFFTTAAdapter(
        module.RFFTTAConfig(
            rff_dim=rff_dim,
            hidden_dims=[128, 128],
            regressor_epochs=epochs,
            batch_size=128,
            tta_steps=100,
            seed=SEED,
        ),
        device="cpu",
    )
    started = time.perf_counter()
    adapter.fit(x.numpy(), summary.numpy(), calibration_split=0.10, alpha=0.05)
    return adapter, time.perf_counter() - started, float(adapter.regressor_losses[-1])


def oup_test_data(cfg: OUPConfig, eps: float, n: int, seed: int):
    prior_gen = torch.Generator().manual_seed(seed)
    clean_gen = torch.Generator().manual_seed(seed + 1)
    contam_gen = torch.Generator().manual_seed(seed + 2)
    mask_gen = torch.Generator().manual_seed(seed + 999)
    theta = sample_oup_prior(n, prior_gen)
    clean = simulate_oup(theta, cfg, clean_gen)
    observed = clean.clone()
    k = int(eps * cfg.n_trajectories)
    if k:
        contam_theta = torch.tensor([[-0.5, 1.0]]).expand(n, -1)
        contaminated = simulate_oup(contam_theta, cfg, contam_gen, variance_scale=0.5)
        for i in range(n):
            idx = torch.randperm(cfg.n_trajectories, generator=mask_gen)[:k]
            observed[i, idx] = contaminated[i, idx]
    return theta, clean, observed


def evaluate_oup_neural(
    model: nn.Module,
    stats: dict[str, torch.Tensor],
    adapter,
    cfg: OUPConfig,
    test_n: int,
    out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for eps in (0.0, 0.1, 0.2, 0.3, 0.4):
        theta, clean, observed = oup_test_data(cfg, eps, test_n, SEED + 100)
        s_clean, s_observed = oup_summary(clean), oup_summary(observed)
        for i in range(test_n):
            started = time.perf_counter()
            result = adapter.adapt(s_observed[i], observed[i].numpy())
            adapt_ms = 1000 * (time.perf_counter() - started)
            s_mds = result["best_s"].reshape(-1)
            base_mean = posterior_mean(model, stats, s_observed[i : i + 1])[0]
            mds_mean = posterior_mean(model, stats, s_mds.unsqueeze(0))[0]
            base_error = float(torch.sqrt(torch.mean((base_mean - theta[i]) ** 2)))
            mds_error = float(torch.sqrt(torch.mean((mds_mean - theta[i]) ** 2)))
            rows.append(
                {
                    "epsilon": eps,
                    "sample": i,
                    "theta_1": float(theta[i, 0]),
                    "theta_2_log": float(theta[i, 1]),
                    "summary_clean": json.dumps(s_clean[i].tolist()),
                    "summary_observed": json.dumps(s_observed[i].tolist()),
                    "summary_mds": json.dumps(s_mds.tolist()),
                    "npe_mean_baseline": json.dumps(base_mean.tolist()),
                    "npe_mean_mds": json.dumps(mds_mean.tolist()),
                    "rmse_baseline": base_error,
                    "rmse_mds": mds_error,
                    "gate_passed": bool(result["gate_passed"]),
                    "lbfgs_evaluations": int(result["n_steps"]),
                    "adapt_ms": adapt_ms,
                }
            )
    trials = pd.DataFrame(rows)
    trials.to_csv(out / "neural_oup_trials.csv", index=False, float_format="%.12g")
    aggregate = []
    for eps, group in trials.groupby("epsilon", sort=True):
        base = group["rmse_baseline"].to_numpy()
        mds = group["rmse_mds"].to_numpy()
        delta = base - mds
        base_rms = float(np.sqrt(np.mean(base**2)))
        mds_rms = float(np.sqrt(np.mean(mds**2)))
        p = 1.0 if np.allclose(delta, 0) else float(ttest_rel(base, mds).pvalue)
        aggregate.append(
            {
                "epsilon": float(eps),
                "trials": len(group),
                "neural_npe_baseline_rmse": base_rms,
                "neural_npe_mds_rmse": mds_rms,
                "rmse_reduction_pct": 100 * (base_rms - mds_rms) / max(base_rms, 1e-12),
                "paired_win_rate": float(np.mean(mds < base)),
                "paired_ttest_pvalue": p,
                "gate_pass_rate": float(group["gate_passed"].mean()),
                "median_adapt_ms": float(group["adapt_ms"].median()),
                "p95_adapt_ms": float(group["adapt_ms"].quantile(0.95)),
            }
        )
    agg = pd.DataFrame(aggregate)
    agg.to_csv(out / "neural_oup_aggregate.csv", index=False, float_format="%.12g")
    return trials, agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    smoke = args.smoke
    out = args.output or ROOT / "outputs" / ("neural_smoke" if smoke else "neural_upgrade")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    params = {
        "gaussian_train": 1_000 if smoke else 20_000,
        "gaussian_epochs": 25 if smoke else 200,
        "oup_train": 1_000 if smoke else 10_000,
        "oup_epochs": 30 if smoke else 240,
        "oup_rff_train": 100 if smoke else 500,
        "oup_rff_dim": 64 if smoke else 512,
        "oup_rff_epochs": 5 if smoke else 60,
        "oup_test": 4 if smoke else 50,
        "oup_trajectories": 20 if smoke else 100,
    }
    started = time.perf_counter()

    gs, gt = gaussian_training_data(params["gaussian_train"], SEED)
    gaussian_model, gaussian_stats, gaussian_fit = fit_neural_posterior(
        gs,
        gt,
        hidden=64,
        epochs=params["gaussian_epochs"],
        patience=30,
        batch_size=256,
        lr=5e-4,
        seed=SEED,
        checkpoint=out / "gaussian_neural_posterior.pt",
        curve_path=out / "gaussian_neural_training.csv",
    )
    gaussian_hash_before = tensor_state_hash(gaussian_model)
    _, gaussian_agg = evaluate_gaussian_neural(gaussian_model, gaussian_stats, out)
    gaussian_hash_after = tensor_state_hash(gaussian_model)

    oup_cfg = OUPConfig(n_trajectories=params["oup_trajectories"])
    osum, otheta = generate_oup_training_summaries(params["oup_train"], oup_cfg, SEED + 20)
    oup_model, oup_stats, oup_fit = fit_neural_posterior(
        osum,
        otheta,
        hidden=128,
        epochs=params["oup_epochs"],
        patience=35,
        batch_size=256,
        lr=5e-4,
        seed=SEED + 21,
        checkpoint=out / "oup_neural_posterior.pt",
        curve_path=out / "oup_neural_training.csv",
    )
    oup_hash_before = tensor_state_hash(oup_model)
    adapter, rff_fit_seconds, rff_final_loss = fit_oup_rff(
        oup_cfg, params["oup_rff_train"], params["oup_rff_dim"], params["oup_rff_epochs"]
    )
    _, oup_agg = evaluate_oup_neural(
        oup_model, oup_stats, adapter, oup_cfg, params["oup_test"], out
    )
    oup_hash_after = tensor_state_hash(oup_model)

    g_contam = gaussian_agg.query("epsilon >= 0.1 and epsilon <= 0.4")
    o_contam = oup_agg.query("epsilon >= 0.1 and epsilon <= 0.4")
    gates = {
        "actual_neural_posteriors": gaussian_fit.parameter_count > 1_000 and oup_fit.parameter_count > 1_000,
        "frozen_gaussian_npe": gaussian_hash_before == gaussian_hash_after,
        "frozen_oup_npe": oup_hash_before == oup_hash_after,
        "gaussian_clean_calibration": float(gaussian_agg.iloc[0]["neural_npe_baseline_rmse"]) < 0.20,
        "gaussian_all_contamination_levels_improve": bool((g_contam["rmse_reduction_pct"] > 0).all()),
        "gaussian_mean_robustness_gain_50pct": float(g_contam["rmse_reduction_pct"].mean()) >= 50,
        "oup_clean_npe_better_than_prior_scale": float(oup_agg.iloc[0]["neural_npe_baseline_rmse"]) < 0.75,
        "oup_majority_contamination_levels_improve": int((o_contam["rmse_reduction_pct"] > 0).sum()) >= 3,
        "oup_mean_contamination_gain_positive": float(o_contam["rmse_reduction_pct"].mean()) > 0,
        "paper_scale_rff_dimension": params["oup_rff_dim"] == 512 if not smoke else True,
        "official_oup_dimensions": (
            oup_cfg.n_timesteps == 25 and oup_cfg.n_trajectories == 100
        ) if not smoke else True,
        "cpu_only": True,
    }
    summary = {
        "run": "smoke" if smoke else "prospectively_frozen_full",
        "paper": "Minimum Distance Summaries for Robust Neural Posterior Estimation",
        "openreview_id": "lq8fNVME8v",
        "protocol_sha256": sha256(PROTOCOL) if PROTOCOL.exists() else None,
        "parameters": params,
        "source_hashes": {
            "official_rff": sha256(OFFICIAL_RFF),
            "official_oup": sha256(OFFICIAL_OUP),
        },
        "gaussian_neural_posterior": asdict(gaussian_fit),
        "oup_neural_posterior": asdict(oup_fit),
        "oup_official_rff": {
            "dimension": params["oup_rff_dim"],
            "fit_seconds": rff_fit_seconds,
            "final_regressor_mse": rff_final_loss,
            "calibrated_tau": float(adapter.calibrated_tau),
            "gamma": float(adapter.gamma),
        },
        "headline": {
            "gaussian_mean_rmse_reduction_pct_eps_0p1_to_0p4": float(g_contam["rmse_reduction_pct"].mean()),
            "oup_mean_rmse_reduction_pct_eps_0p1_to_0p4": float(o_contam["rmse_reduction_pct"].mean()),
            "oup_levels_improved": int((o_contam["rmse_reduction_pct"] > 0).sum()),
            "oup_contamination_levels": len(o_contam),
            "oup_median_adaptation_ms": float(o_contam["median_adapt_ms"].median()),
        },
        "gates": gates,
        "all_frozen_gates_pass": all(gates.values()),
        "runtime_seconds": time.perf_counter() - started,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_used": False,
            "mps_used": False,
        },
    }
    write_json(out / "summary.json", summary)
    checksum_paths = sorted(path for path in out.iterdir() if path.is_file())
    (out / "CHECKSUMS.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths if path.name != "CHECKSUMS.sha256")
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not smoke and not summary["all_frozen_gates_pass"]:
        raise SystemExit("one or more prospectively frozen neural-upgrade gates failed; outputs retained")


if __name__ == "__main__":
    main()

````


````output
Calibrated tau=0.033690 (alpha=0.05, n_calib=50)
  Loss stats: mean=0.018227, std=0.009003
{
  "all_frozen_gates_pass": true,
  "environment": {
    "cuda_visible_devices": "",
    "device": "cpu",
    "gpu_used": false,
    "mps_used": false,
    "platform": "Linux-7.0.9-arch2-1-x86_64-with-glibc2.43",
    "python": "3.12.13 (main, Jun 23 2026, 15:18:55) [Clang 22.1.3 ]",
    "torch": "2.12.0+cu130"
  },
  "gates": {
    "actual_neural_posteriors": true,
    "cpu_only": true,
    "frozen_gaussian_npe": true,
    "frozen_oup_npe": true,
    "gaussian_all_contamination_levels_improve": true,
    "gaussian_clean_calibration": true,
    "gaussian_mean_robustness_gain_50pct": true,
    "official_oup_dimensions": true,
    "oup_clean_npe_better_than_prior_scale": true,
    "oup_majority_contamination_levels_improve": true,
    "oup_mean_contamination_gain_positive": true,
    "paper_scale_rff_dimension": true
  },
  "gaussian_neural_posterior": {
    "best_epoch": 74,
    "best_val_nll": -2.472168207168579,
    "epochs_ran": 104,
    "model_sha256": "38161a0ab9d3dfc5b0893964094d4c062298d16d41a62890b03e8ce657fba5aa",
    "parameter_count": 4418,
    "summary_mean": [
      -0.00015641245408914983
    ],
    "summary_std": [
      1.997832179069519
    ],
    "theta_mean": [
      -0.00041640427662059665
    ],
    "theta_std": [
      1.9958429336547852
    ],
    "train_seconds": 9.3037848288659
  },
  "headline": {
    "gaussian_mean_rmse_reduction_pct_eps_0p1_to_0p4": 91.46203643341545,
    "oup_contamination_levels": 4,
    "oup_levels_improved": 4,
    "oup_mean_rmse_reduction_pct_eps_0p1_to_0p4": 69.10760881281861,
    "oup_median_adaptation_ms": 134.52396827051416
  },
  "openreview_id": "lq8fNVME8v",
  "oup_neural_posterior": {
    "best_epoch": 141,
    "best_val_nll": -6.458736419677734,
    "epochs_ran": 176,
    "model_sha256": "4e856211d806c14961078b3b8e0469f5a1db151d9afebdb785c1b4b91407ff50",
    "parameter_count": 17540,
    "summary_mean": [
      2.8321738243103027,
      0.028982825577259064,
      0.8597376942634583
    ],
    "summary_std": [
      2.474567413330078,
      0.04438817873597145,
      0.13911762833595276
    ],
    "theta_mean": [
      0.9985845685005188,
      0.00044157859520055354
    ],
    "theta_std": [
      0.5776110291481018,
      1.1630735397338867
    ],
    "train_seconds": 9.711613974999636
  },
  "oup_official_rff": {
    "calibrated_tau": 0.033690001256763934,
    "dimension": 512,
    "final_regressor_mse": 2.8292737624724396e-05,
    "fit_seconds": 1.6122340760193765,
    "gamma": 0.003818355966359377
  },
  "paper": "Minimum Distance Summaries for Robust Neural Posterior Estimation",
  "parameters": {
    "gaussian_epochs": 200,
    "gaussian_train": 20000,
    "oup_epochs": 240,
    "oup_rff_dim": 512,
    "oup_rff_epochs": 60,
    "oup_rff_train": 500,
    "oup_test": 50,
    "oup_train": 10000,
    "oup_trajectories": 100
  },
  "protocol_sha256": "ede45cf04931ea5a2297228faa9bf1a42833d2f7ccda2df3a31b7e3e26cf1a02",
  "run": "prospectively_frozen_full",
  "runtime_seconds": 52.048699978971854,
  "source_hashes": {
    "official_oup": "a256e686fff9fcc94c2710c48de279e25568420eb6acd495d962753d2fdf0a50",
    "official_rff": "0a7e8a2261c9ab9f8b327c01335dd7ff9cb7d5efd48abcf1980bf9c125d329b9"
  }
}

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_3da18dd63e8f", "created_at": "2026-07-16T15:53:33+00:00", "title": "Artifact: neural_oup_trials.csv", "path": "outputs/neural_upgrade/neural_oup_trials.csv", "size": 91529, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/neural_upgrade/neural_oup_trials.csv` · dataset · 91.5 kB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/neural_upgrade/neural_oup_trials.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_1c97b06d7c6a", "created_at": "2026-07-16T15:53:33+00:00", "title": "Artifact: oup_neural_posterior.pt", "path": "outputs/neural_upgrade/oup_neural_posterior.pt", "size": 74581, "artifact_type": "model", "auto": true}
-->
**📦 Artifact** `outputs/neural_upgrade/oup_neural_posterior.pt` · model · 74.6 kB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/neural_upgrade/oup_neural_posterior.pt


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_e2c63c4b051b", "created_at": "2026-07-16T15:53:33+00:00", "title": "Artifact: neural_gaussian_trials.csv", "path": "outputs/neural_upgrade/neural_gaussian_trials.csv", "size": 34437, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/neural_upgrade/neural_gaussian_trials.csv` · dataset · 34.4 kB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/neural_upgrade/neural_gaussian_trials.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_12ca445ec6d3", "created_at": "2026-07-16T15:53:33+00:00", "title": "Artifact: gaussian_neural_posterior.pt", "path": "outputs/neural_upgrade/gaussian_neural_posterior.pt", "size": 22245, "artifact_type": "model", "auto": true}
-->
**📦 Artifact** `outputs/neural_upgrade/gaussian_neural_posterior.pt` · model · 22.2 kB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/neural_upgrade/gaussian_neural_posterior.pt


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_fc1447577089", "created_at": "2026-07-16T15:53:33+00:00", "title": "Artifact: oup_neural_training.csv", "path": "outputs/neural_upgrade/oup_neural_training.csv", "size": 5873, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/neural_upgrade/oup_neural_training.csv` · dataset · 5.9 kB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/neural_upgrade/oup_neural_training.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_00d571145e31", "created_at": "2026-07-16T15:53:34+00:00", "title": "Artifact: gaussian_neural_training.csv", "path": "outputs/neural_upgrade/gaussian_neural_training.csv", "size": 3434, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/neural_upgrade/gaussian_neural_training.csv` · dataset · 3.4 kB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/neural_upgrade/gaussian_neural_training.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_039b24676ad8", "created_at": "2026-07-16T15:53:34+00:00", "title": "Artifact: neural_oup_aggregate.csv", "path": "outputs/neural_upgrade/neural_oup_aggregate.csv", "size": 671, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/neural_upgrade/neural_oup_aggregate.csv` · dataset · 671 B

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/neural_upgrade/neural_oup_aggregate.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_7f7965d4c3df", "created_at": "2026-07-16T15:53:34+00:00", "title": "Artifact: neural_gaussian_aggregate.csv", "path": "outputs/neural_upgrade/neural_gaussian_aggregate.csv", "size": 553, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/neural_upgrade/neural_gaussian_aggregate.csv` · dataset · 553 B

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/neural_upgrade/neural_gaussian_aggregate.csv


---
<!-- trackio-cell
{"type": "code", "id": "cell_1725b86889e1", "created_at": "2026-07-16T15:53:34+00:00", "title": "Integrate neural evidence", "command": [".venv/bin/python", "reproduction/integrate_neural_upgrade.py"], "exit_code": 0, "duration_s": 0.034}
-->
````bash
$ .venv/bin/python reproduction/integrate_neural_upgrade.py
````

exit 0 · 0.0s


````python title=integrate_neural_upgrade.py
#!/usr/bin/env python3
"""Fail-closed integration of the prospectively frozen neural/OUP upgrade."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "neural_upgrade"
OUT = ROOT / "outputs" / "full"


def main() -> None:
    upgrade = json.loads((SOURCE / "summary.json").read_text())
    if upgrade["run"] != "prospectively_frozen_full" or not upgrade["all_frozen_gates_pass"]:
        raise AssertionError("refusing to integrate a non-full or failed neural upgrade")
    if upgrade["parameters"]["oup_rff_dim"] != 512:
        raise AssertionError("paper-scale RFF dimension not retained")
    if upgrade["parameters"]["oup_trajectories"] != 100:
        raise AssertionError("official OUP trajectory dimension not retained")

    mapping = {
        "summary.json": "neural_upgrade_summary.json",
        "gaussian_neural_posterior.pt": "gaussian_neural_posterior.pt",
        "oup_neural_posterior.pt": "oup_neural_posterior.pt",
        "gaussian_neural_training.csv": "gaussian_neural_training.csv",
        "oup_neural_training.csv": "oup_neural_training.csv",
        "neural_gaussian_trials.csv": "neural_gaussian_trials.csv",
        "neural_gaussian_aggregate.csv": "neural_gaussian_aggregate.csv",
        "neural_oup_trials.csv": "neural_oup_trials.csv",
        "neural_oup_aggregate.csv": "neural_oup_aggregate.csv",
    }
    for source_name, target_name in mapping.items():
        source = SOURCE / source_name
        if not source.is_file():
            raise AssertionError(f"missing upgrade file: {source}")
        shutil.copy2(source, OUT / target_name)

    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["claims"]["claim_1_plugin_mds_decoupled_from_pretrained_npe"] = {
        "verdict": "verified_with_two_actual_frozen_neural_posteriors",
        "evidence": (
            "Two trained conditional-density NPEs (Gaussian and OUP; 4,418 and 17,540 "
            "parameters) are hash-identical before and after every adaptation. The same "
            "frozen NPE is queried with ordinary versus official-RFF MDS summaries."
        ),
    }
    summary["claims"]["claim_3_robustness_gain_minimal_overhead"] = {
        "verdict": "verified_with_actual_neural_posteriors_across_gaussian_and_oup",
        "evidence": {
            "gaussian_mean_rmse_reduction_pct_eps_0p1_to_0p4": upgrade["headline"][
                "gaussian_mean_rmse_reduction_pct_eps_0p1_to_0p4"
            ],
            "oup_mean_rmse_reduction_pct_eps_0p1_to_0p4": upgrade["headline"][
                "oup_mean_rmse_reduction_pct_eps_0p1_to_0p4"
            ],
            "oup_levels_improved": upgrade["headline"]["oup_levels_improved"],
            "oup_contamination_levels": upgrade["headline"]["oup_contamination_levels"],
            "oup_median_adaptation_ms": upgrade["headline"]["oup_median_adaptation_ms"],
            "all_frozen_gates_pass": upgrade["all_frozen_gates_pass"],
        },
    }
    summary["neural_upgrade"] = upgrade
    summary["wall_seconds_total"] = float(summary["wall_seconds_total"]) + float(
        upgrade["runtime_seconds"]
    )
    for name in mapping.values():
        if name not in summary["raw_files"]:
            summary["raw_files"].append(name)
    summary["raw_files"].sort()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "integrated": True,
                "copied_files": len(mapping),
                "gaussian_gain_pct": upgrade["headline"][
                    "gaussian_mean_rmse_reduction_pct_eps_0p1_to_0p4"
                ],
                "oup_gain_pct": upgrade["headline"][
                    "oup_mean_rmse_reduction_pct_eps_0p1_to_0p4"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

````


````output
{"copied_files": 9, "gaussian_gain_pct": 91.46203643341545, "integrated": true, "oup_gain_pct": 69.10760881281861}

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_4a91f105be11", "created_at": "2026-07-16T15:53:34+00:00", "title": "Artifact: neural_oup_trials.csv", "path": "outputs/full/neural_oup_trials.csv", "size": 91529, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/full/neural_oup_trials.csv` · dataset · 91.5 kB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/full/neural_oup_trials.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_6ba4f79a90b7", "created_at": "2026-07-16T15:53:35+00:00", "title": "Artifact: oup_neural_posterior.pt", "path": "outputs/full/oup_neural_posterior.pt", "size": 74581, "artifact_type": "model", "auto": true}
-->
**📦 Artifact** `outputs/full/oup_neural_posterior.pt` · model · 74.6 kB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/full/oup_neural_posterior.pt


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_944ab14e8c80", "created_at": "2026-07-16T15:53:35+00:00", "title": "Artifact: neural_gaussian_trials.csv", "path": "outputs/full/neural_gaussian_trials.csv", "size": 34437, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/full/neural_gaussian_trials.csv` · dataset · 34.4 kB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/full/neural_gaussian_trials.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_0726e9d8f10b", "created_at": "2026-07-16T15:53:35+00:00", "title": "Artifact: gaussian_neural_posterior.pt", "path": "outputs/full/gaussian_neural_posterior.pt", "size": 22245, "artifact_type": "model", "auto": true}
-->
**📦 Artifact** `outputs/full/gaussian_neural_posterior.pt` · model · 22.2 kB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/full/gaussian_neural_posterior.pt


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_e4f398e51dbb", "created_at": "2026-07-16T15:53:35+00:00", "title": "Artifact: oup_neural_training.csv", "path": "outputs/full/oup_neural_training.csv", "size": 5873, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/full/oup_neural_training.csv` · dataset · 5.9 kB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/full/oup_neural_training.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_2cbb7e82bec5", "created_at": "2026-07-16T15:53:35+00:00", "title": "Artifact: gaussian_neural_training.csv", "path": "outputs/full/gaussian_neural_training.csv", "size": 3434, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/full/gaussian_neural_training.csv` · dataset · 3.4 kB

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/full/gaussian_neural_training.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_21295ab00482", "created_at": "2026-07-16T15:53:35+00:00", "title": "Artifact: neural_oup_aggregate.csv", "path": "outputs/full/neural_oup_aggregate.csv", "size": 671, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/full/neural_oup_aggregate.csv` · dataset · 671 B

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/full/neural_oup_aggregate.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_6edc131093d7", "created_at": "2026-07-16T15:53:35+00:00", "title": "Artifact: neural_gaussian_aggregate.csv", "path": "outputs/full/neural_gaussian_aggregate.csv", "size": 553, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/full/neural_gaussian_aggregate.csv` · dataset · 553 B

https://huggingface.co/buckets/DineshAI/lq8fNVME8v-artifacts#logbook-files/outputs/full/neural_gaussian_aggregate.csv


---
<!-- trackio-cell
{"type": "code", "id": "cell_f36e52393ff6", "created_at": "2026-07-16T15:53:36+00:00", "title": "Fail-closed 16-test suite", "command": ["bash", "-lc", "cd reproduction && ../.venv/bin/python -m unittest -v test_reproduction.py"], "exit_code": 0, "duration_s": 0.36}
-->
````bash
$ bash -lc 'cd reproduction && ../.venv/bin/python -m unittest -v test_reproduction.py'
````

exit 0 · 0.4s


````output
test_01_exact_challenge_identity_and_four_claims (test_reproduction.TestMDSReproduction.test_01_exact_challenge_identity_and_four_claims) ... ok
test_02_primary_sources_and_commit_are_hash_pinned (test_reproduction.TestMDSReproduction.test_02_primary_sources_and_commit_are_hash_pinned) ... ok
test_03_official_algorithm_signatures_all_present (test_reproduction.TestMDSReproduction.test_03_official_algorithm_signatures_all_present) ... ok
test_04_all_300_trials_present_and_finite (test_reproduction.TestMDSReproduction.test_04_all_300_trials_present_and_finite) ... ok
test_05_clean_calibration_gate_preserves_baseline (test_reproduction.TestMDSReproduction.test_05_clean_calibration_gate_preserves_baseline) ... ok
test_06_substantial_robustness_gain_on_prespecified_range (test_reproduction.TestMDSReproduction.test_06_substantial_robustness_gain_on_prespecified_range) ... ok
test_07_test_time_adaptation_is_millisecond_scale (test_reproduction.TestMDSReproduction.test_07_test_time_adaptation_is_millisecond_scale) ... ok
test_08_exact_mmd_reference_and_severe_limit_disclosed (test_reproduction.TestMDSReproduction.test_08_exact_mmd_reference_and_severe_limit_disclosed) ... ok
test_09_bounded_influence_special_case (test_reproduction.TestMDSReproduction.test_09_bounded_influence_special_case) ... ok
test_10_consistency_special_case_contracts (test_reproduction.TestMDSReproduction.test_10_consistency_special_case_contracts) ... ok
test_11_theory_scope_boundaries_are_explicit (test_reproduction.TestMDSReproduction.test_11_theory_scope_boundaries_are_explicit) ... ok
test_12_dimension_general_proof_certificate (test_reproduction.TestMDSReproduction.test_12_dimension_general_proof_certificate) ... ok
test_13_pinned_cpu_environment_is_documented (test_reproduction.TestMDSReproduction.test_13_pinned_cpu_environment_is_documented) ... ok
test_14_actual_neural_posteriors_are_frozen (test_reproduction.TestMDSReproduction.test_14_actual_neural_posteriors_are_frozen) ... ok
test_15_gaussian_neural_npe_robustness (test_reproduction.TestMDSReproduction.test_15_gaussian_neural_npe_robustness) ... ok
test_16_oup_neural_npe_second_mechanism (test_reproduction.TestMDSReproduction.test_16_oup_neural_npe_second_mechanism) ... ok

----------------------------------------------------------------------
Ran 16 tests in 0.028s

OK

````
