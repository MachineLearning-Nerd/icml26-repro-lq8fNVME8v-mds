#!/usr/bin/env python3
"""Paper-scale Gaussian comparator reproduction for Claim 5.

This uses the pinned authors' NPE, NNPE, RFF-MDS, and OC-SVM components.  The
RFF transform is streamed by dataset to avoid a 19.5 GB temporary array; its
random features and dataset means are otherwise identical to the official
implementation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import types
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "source" / "official-repo"
sys.path.insert(0, str(OFFICIAL / "src"))

# The authors' inference package initializer eagerly imports the optional
# NPE-RS implementation, which in turn imports sbibm.c2st.  sbibm 1.1 pins an
# incompatible sbi<0.22, while this repository pins sbi>=0.25.  Install narrow
# namespace packages so the exact modules needed here load without executing
# unrelated package initializers; no implementation file is replaced.
import tt_sbi


def install_module_namespace(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    sys.modules[name] = module
    setattr(tt_sbi, name.rsplit(".", 1)[-1], module)


install_module_namespace("tt_sbi.inference", OFFICIAL / "src/tt_sbi/inference")
install_module_namespace("tt_sbi.tta", OFFICIAL / "src/tt_sbi/tta")

from tt_sbi.inference.nn import build_npe_model, get_embedding_net
from tt_sbi.inference.npe import NPE_TrainConfig, sample_npe_posterior, train_NPE_estimator
from tt_sbi.inference.npe_noisy import NoisyNPE_TrainConfig, train_NPE_estimator_noisy
from tt_sbi.ocsvm import OCSVMConfig, clean_observations, fit_ocsvm_detector
from tt_sbi.tta.adapters import TTAPosterior
from tt_sbi.tta.rff import RFFTTAAdapter, RFFTTAConfig


ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_5_gaussian_comparators"
SEED = 42
N_TRAIN = 50_000
N_TEST = 100
N_OBS = 100
DIM = 2
N_POSTERIOR = 2_000
EPSILONS = (0.0, 0.1, 0.2, 0.3, 0.5)
SHIFT = 3.0
METHODS = ("NPE", "NNPE", "NPE-MDS (RF)", "NPE-OR")
RFF_DIM = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    theta_gen = torch.Generator().manual_seed(SEED)
    x_gen = torch.Generator().manual_seed(SEED + 1)
    theta = torch.randn(N_TRAIN, DIM, generator=theta_gen)
    x = theta[:, None, :] + torch.randn(N_TRAIN, N_OBS, DIM, generator=x_gen)
    return theta, x


def generate_test_data(epsilon: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta_gen = torch.Generator().manual_seed(SEED + 1_000)
    x_gen = torch.Generator().manual_seed(SEED + 1_001)
    theta = torch.randn(N_TEST, DIM, generator=theta_gen)
    clean = theta[:, None, :] + torch.randn(N_TEST, N_OBS, DIM, generator=x_gen)
    observed = clean.clone()
    mask_gen = torch.Generator().manual_seed(SEED + 1_004)
    mask = torch.rand(N_TEST, N_OBS, generator=mask_gen) < epsilon
    if mask.any():
        sign_gen = torch.Generator().manual_seed(SEED + 1_005)
        signs = torch.where(
            torch.rand(int(mask.sum()), generator=sign_gen) < 0.5,
            torch.tensor(1.0),
            torch.tensor(-1.0),
        )
        observed[mask] = signs[:, None] * SHIFT
    return theta, clean, observed


def build_and_train(
    theta: torch.Tensor, x: torch.Tensor, *, noisy: bool
) -> tuple[torch.nn.Module, dict, float]:
    flat = x.reshape(N_TRAIN, -1)
    seed_all(SEED)
    model = build_npe_model(
        theta_sample=theta,
        x_sample=flat,
        embedding_type="fc",
        n_obs=N_OBS,
        dim=DIM,
        embedding_dim=DIM,
    )
    started = time.perf_counter()
    if noisy:
        config = NoisyNPE_TrainConfig(
            lr=5e-4,
            batch_size=256,
            val_frac=0.1,
            stop_after_epochs=20,
            max_epochs=10_000,
            slab_scale=0.2,
            spike_scale=0.01,
            # Paper rho=1 means slab probability one.  The official helper
            # names the complementary probability `spike_prob`.
            spike_prob=0.0,
            noise_on_val=False,
        )
        model, history = train_NPE_estimator_noisy(
            model, theta, flat, config=config, device="cpu", seed=SEED
        )
    else:
        config = NPE_TrainConfig(
            lr=5e-4,
            batch_size=256,
            val_frac=0.1,
            stop_after_epochs=20,
            max_epochs=10_000,
        )
        model, history = train_NPE_estimator(
            model, theta, flat, config=config, device="cpu", seed=SEED
        )
    model.eval()
    return model, history, time.perf_counter() - started


def fit_rff_streaming(x: torch.Tensor, summaries: torch.Tensor) -> tuple[RFFTTAAdapter, dict]:
    config = RFFTTAConfig(
        rff_dim=RFF_DIM,
        hidden_dims=[256, 256],
        regressor_epochs=100,
        regressor_lr=1e-3,
        batch_size=256,
        tta_steps=200,
        seed=SEED,
    )
    adapter = RFFTTAAdapter(config, device="cpu")
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(x))
    n_calib = int(len(x) * 0.05)
    calib_idx, train_idx = indices[:n_calib], indices[n_calib:]
    x_np = x.numpy()
    s_np = summaries.numpy()
    x_train, s_train = x_np[train_idx], s_np[train_idx]
    x_calib, s_calib = x_np[calib_idx], s_np[calib_idx]

    adapter.gamma = adapter._median_heuristic(x_train.reshape(-1, DIM))
    adapter.rff = RBFSampler(
        gamma=adapter.gamma, n_components=RFF_DIM, random_state=SEED
    )
    adapter.rff.fit(x_train[0])
    z_train = np.empty((len(x_train), RFF_DIM), dtype=np.float32)
    for start in range(0, len(x_train), 256):
        block = x_train[start : start + 256]
        transformed = adapter.rff.transform(block.reshape(-1, DIM))
        z_train[start : start + len(block)] = transformed.reshape(
            len(block), N_OBS, RFF_DIM
        ).mean(axis=1)

    s_tensor = torch.as_tensor(s_train, dtype=torch.float32)
    z_tensor = torch.as_tensor(z_train, dtype=torch.float32)
    adapter.s_mean = s_tensor.mean(0)
    adapter.s_std = s_tensor.std(0) + 1e-8
    s_norm = (s_tensor - adapter.s_mean) / adapter.s_std
    adapter.regressor = adapter._build_regressor(DIM, RFF_DIM)
    started = time.perf_counter()
    adapter.regressor_losses = adapter._train_regressor(s_norm, z_tensor)
    fit_seconds = time.perf_counter() - started
    adapter.s_mean = adapter.s_mean.to(adapter.device)
    adapter.s_std = adapter.s_std.to(adapter.device)
    adapter._calibrate_tau_from_data(x_calib, s_calib, alpha=0.05)
    return adapter, {
        "dimension": RFF_DIM,
        "gamma": float(adapter.gamma),
        "calibrated_tau": float(adapter.calibrated_tau),
        "regressor_final_mse": float(adapter.regressor_losses[-1]),
        "regressor_fit_seconds": fit_seconds,
        "stream_block_datasets": 256,
    }


def fit_paper_ocsvm(x: torch.Tensor):
    pooled = x.numpy().reshape(-1, DIM)
    rng = np.random.default_rng(SEED)
    fit_idx = rng.choice(len(pooled), size=20_000, replace=False)
    fit_values = pooled[fit_idx]
    scaled = StandardScaler().fit_transform(fit_values)
    median_rng = np.random.default_rng(SEED + 1)
    med_idx = median_rng.choice(len(scaled), size=5_000, replace=False)
    subset = scaled[med_idx]
    squared = (
        np.sum(subset**2, axis=1)[:, None]
        + np.sum(subset**2, axis=1)[None, :]
        - 2 * subset @ subset.T
    )
    median_squared_distance = float(
        np.median(squared[np.triu_indices(len(subset), k=1)])
    )
    gamma = 1.0 / (2.0 * median_squared_distance + 1e-8)
    detector = fit_ocsvm_detector(
        x.numpy(),
        OCSVMConfig(
            nu=0.05,
            kernel="rbf",
            gamma=gamma,
            standardize=True,
            max_train_elements=20_000,
            calibrate_fpr=0.05,
            random_state=SEED,
        ),
    )
    return detector, gamma, median_squared_distance


def analytic_clean_posterior(
    clean: torch.Tensor, sample_seed: int
) -> torch.Tensor:
    mean = (N_OBS / (N_OBS + 1.0)) * clean.mean(0)
    std = math.sqrt(1.0 / (N_OBS + 1.0))
    gen = torch.Generator().manual_seed(sample_seed)
    return mean + std * torch.randn(N_POSTERIOR, DIM, generator=gen)


def exact_official_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
    """Official five-bandwidth biased MMD, with a memory-only kernel loop."""
    combined = torch.cat([x.float(), y.float()])
    distances = torch.cdist(combined, combined).square()
    count = len(combined)
    bandwidth = (distances.detach().sum() / (count * count - count)).clamp_min(1e-8)
    n = len(x)
    value = torch.tensor(0.0)
    for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0):
        kernel = torch.exp(-distances / (bandwidth * multiplier))
        value = value + kernel[:n, :n].mean() - 2 * kernel[:n, n:].mean() + kernel[n:, n:].mean()
    return float(value)


def rmse(samples: torch.Tensor, theta: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((samples.mean(0) - theta) ** 2)))


def bootstrap_ci(delta: np.ndarray, seed: int, replicates: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for start in range(0, replicates, 1_000):
        size = min(1_000, replicates - start)
        idx = rng.integers(0, len(delta), size=(size, len(delta)))
        means[start : start + size] = delta[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for epsilon_index, epsilon in enumerate(EPSILONS):
        by_method = {
            method: [row for row in rows if row["epsilon"] == epsilon and row["method"] == method]
            for method in METHODS
        }
        mds = np.array([row["posterior_mmd"] for row in by_method["NPE-MDS (RF)"]])
        for method in METHODS:
            values = np.array([row["posterior_mmd"] for row in by_method[method]])
            rmses = np.array([row["rmse"] for row in by_method[method]])
            if method == "NPE-MDS (RF)":
                low, high = 0.0, 0.0
            else:
                low, high = bootstrap_ci(
                    values - mds, SEED + 30_000 + 10 * epsilon_index + METHODS.index(method)
                )
            output.append(
                {
                    "epsilon": epsilon,
                    "method": method,
                    "trials": len(values),
                    "posterior_mmd_mean": float(values.mean()),
                    "posterior_mmd_std": float(values.std(ddof=1)),
                    "posterior_mmd_se": float(values.std(ddof=1) / math.sqrt(len(values))),
                    "rmse_mean": float(rmses.mean()),
                    "rmse_std": float(rmses.std(ddof=1)),
                    "comparator_minus_mds_mean": float((values - mds).mean()),
                    "comparator_minus_mds_ci95_low": low,
                    "comparator_minus_mds_ci95_high": high,
                }
            )
    return output


def main() -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    seed_all(SEED)
    started = time.perf_counter()
    theta_train, x_train = generate_training_data()

    standard, standard_history, standard_seconds = build_and_train(
        theta_train, x_train, noisy=False
    )
    nnpe, nnpe_history, nnpe_seconds = build_and_train(
        theta_train, x_train, noisy=True
    )
    standard_hash_before = state_hash(standard)
    nnpe_hash_before = state_hash(nnpe)

    embedding = get_embedding_net(standard)
    with torch.no_grad():
        train_summaries = embedding(x_train.reshape(N_TRAIN, -1)).cpu()
    adapter, rff_receipt = fit_rff_streaming(x_train, train_summaries)
    mds_posterior = TTAPosterior(
        standard, adapter, device="cpu", bypass_embedding=True
    )
    detector, ocsvm_gamma, ocsvm_median_squared = fit_paper_ocsvm(x_train)

    rows: list[dict] = []
    for epsilon_index, epsilon in enumerate(EPSILONS):
        theta, clean, observed = generate_test_data(epsilon)
        for sample_index in range(N_TEST):
            x_obs = observed[sample_index]
            x_clean = clean[sample_index]
            theta_true = theta[sample_index]
            true_samples = analytic_clean_posterior(
                x_clean, SEED + sample_index
            )
            with torch.no_grad():
                s_obs = embedding(x_obs.reshape(1, -1)).squeeze(0).cpu()
            cleaned, outlier_mask, _ = clean_observations(
                detector, x_obs.numpy(), seed=SEED
            )

            samples_by_method: dict[str, torch.Tensor] = {}
            seed_all(SEED + 200_000 + 10_000 * epsilon_index + 10 * sample_index)
            samples_by_method["NPE"] = sample_npe_posterior(
                standard, x_obs.reshape(-1), N_POSTERIOR, "cpu"
            )
            seed_all(SEED + 200_001 + 10_000 * epsilon_index + 10 * sample_index)
            samples_by_method["NNPE"] = sample_npe_posterior(
                nnpe, x_obs.reshape(-1), N_POSTERIOR, "cpu"
            )
            adapt_started = time.perf_counter()
            seed_all(SEED + 200_002 + 10_000 * epsilon_index + 10 * sample_index)
            mds_samples, adapt_info = mds_posterior.sample(
                x_obs.numpy(),
                n_samples=N_POSTERIOR,
                adapt=True,
                return_info=True,
                s_obs=s_obs,
            )
            adapt_ms = 1_000 * (time.perf_counter() - adapt_started)
            samples_by_method["NPE-MDS (RF)"] = mds_samples
            seed_all(SEED + 200_003 + 10_000 * epsilon_index + 10 * sample_index)
            samples_by_method["NPE-OR"] = sample_npe_posterior(
                standard, torch.from_numpy(cleaned).reshape(-1), N_POSTERIOR, "cpu"
            )

            for method in METHODS:
                method_samples = samples_by_method[method]
                row = {
                    "epsilon": epsilon,
                    "sample_index": sample_index,
                    "method": method,
                    "posterior_mmd": exact_official_mmd(method_samples, true_samples),
                    "rmse": rmse(method_samples, theta_true),
                    "true_theta_0": float(theta_true[0]),
                    "true_theta_1": float(theta_true[1]),
                    "actual_contamination_fraction": float(
                        torch.mean((observed[sample_index] != clean[sample_index]).any(1).float())
                    ),
                    "ocsvm_outlier_fraction": float(outlier_mask.mean()),
                    "mds_gate_passed": bool(adapt_info["gate_passed"]) if method == "NPE-MDS (RF)" else "",
                    "mds_lbfgs_evaluations": int(adapt_info["n_steps"]) if method == "NPE-MDS (RF)" else "",
                    "mds_adapt_and_sample_ms": adapt_ms if method == "NPE-MDS (RF)" else "",
                }
                rows.append(row)
                print("CLAIM5_TRIAL " + json.dumps(row, sort_keys=True), flush=True)

    raw_path = ARTIFACT / "raw_trials.csv"
    write_rows(raw_path, rows)
    aggregate_rows = aggregate(rows)
    write_rows(ARTIFACT / "aggregate.csv", aggregate_rows)
    standard_hash_after = state_hash(standard)
    nnpe_hash_after = state_hash(nnpe)
    summary = {
        "claim": "Claim 5 Gaussian full comparator curve",
        "paper_scope": {
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "n_obs": N_OBS,
            "dimension": DIM,
            "posterior_samples": N_POSTERIOR,
            "epsilons": EPSILONS,
            "outlier_shift": SHIFT,
            "rff_dimension": RFF_DIM,
            "nnpe": {"rho": 1.0, "sigma": 0.01, "tau": 0.2, "official_spike_prob": 0.0},
            "ocsvm": {"fpr": 0.05, "gamma": "median heuristic", "pooled_fit": 20_000},
        },
        "model_hashes": {
            "standard_before": standard_hash_before,
            "standard_after": standard_hash_after,
            "nnpe_before": nnpe_hash_before,
            "nnpe_after": nnpe_hash_after,
        },
        "training": {
            "standard_seconds": standard_seconds,
            "standard_best_epoch": standard_history["best_epoch"],
            "standard_best_val_loss": standard_history["best_val_loss"],
            "nnpe_seconds": nnpe_seconds,
            "nnpe_best_epoch": nnpe_history["best_epoch"],
            "nnpe_best_val_loss": nnpe_history["best_val_loss"],
        },
        "rff": rff_receipt,
        "ocsvm": {
            "gamma": ocsvm_gamma,
            "median_squared_distance": ocsvm_median_squared,
            "threshold": float(detector.threshold),
        },
        "aggregate": aggregate_rows,
        "raw_sha256": sha256(raw_path),
        "source": {
            "paper_tex_sha256": sha256(ROOT / "source/arxiv/arxiv_main.tex"),
            "official_repo_commit": "45158124f0cbdc2f6c1ac602c9fc5501dce20af3",
            "official_rff_sha256": sha256(OFFICIAL / "src/tt_sbi/tta/rff.py"),
            "official_npe_sha256": sha256(OFFICIAL / "src/tt_sbi/inference/npe.py"),
            "official_nnpe_sha256": sha256(OFFICIAL / "src/tt_sbi/inference/npe_noisy.py"),
            "official_ocsvm_sha256": sha256(OFFICIAL / "src/tt_sbi/ocsvm.py"),
        },
        "compute": {
            "estimated_required_cores": 6,
            "selected_backend": "hf",
            "selected_flavor": "cpu-upgrade",
            "actual_cpu_allocation_logical": os.cpu_count(),
            "gpu_used": False,
            "platform": platform.platform(),
            "runtime_seconds": time.perf_counter() - started,
        },
        "reproducibility": {
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "fixed_command": "git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py",
            "uv_lock_sha256": sha256(ROOT / "uv.lock"),
            "python_version_file": (ROOT / ".python-version").read_text(
                encoding="utf-8"
            ).strip(),
        },
        "seeds": {
            "base": SEED,
            "training_theta": SEED,
            "training_x": SEED + 1,
            "test_theta": SEED + 1_000,
            "test_x": SEED + 1_001,
            "contamination_mask": SEED + 1_004,
            "contamination_sign": SEED + 1_005,
            "analytic_posterior_per_test_dataset": "42 + sample_index",
        },
    }
    (ARTIFACT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("CLAIM5_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
