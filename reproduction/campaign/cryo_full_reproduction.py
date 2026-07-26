#!/usr/bin/env python3
"""Full-scale CPU reproduction of the paper's 1024D Cryo-EM claim."""

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
import tempfile
import time
import types
from pathlib import Path
from types import SimpleNamespace

REQUESTED_THREADS = min(32, os.cpu_count() or 1)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = str(REQUESTED_THREADS)
os.environ["MKL_NUM_THREADS"] = str(REQUESTED_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(REQUESTED_THREADS)

import numpy as np
import torch
from sklearn.kernel_approximation import RBFSampler


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "source" / "official-repo"
CRYOSBI = ROOT / "external" / "cryoSBI"
sys.path.insert(0, str(OFFICIAL / "src"))
sys.path.insert(0, str(CRYOSBI / "src"))

# Avoid the authors' unrelated eager NPE-RS -> sbibm import. The exact modules
# used below are still loaded from the pinned official source tree.
import tt_sbi


def install_module_namespace(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    sys.modules[name] = module
    setattr(tt_sbi, name.rsplit(".", 1)[-1], module)


install_module_namespace("tt_sbi.inference", OFFICIAL / "src/tt_sbi/inference")
install_module_namespace("tt_sbi.tta", OFFICIAL / "src/tt_sbi/tta")

from tt_sbi.inference.nn import build_npe_model
from tt_sbi.inference.npe import (
    NPE_TrainConfig,
    sample_npe_posterior,
    train_NPE_estimator,
)
from tt_sbi.tasks.cyro_em import CryoEMTask
from tt_sbi.tta.adapters import TTAPosterior
from tt_sbi.tta.rff import RFFTTAAdapter, RFFTTAConfig


ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_6_cryo_em"
SEED = 42
N_TRAIN = 15_000
N_RFF_DATASETS = 10_000
N_TEST = 100
N_OBS = 100
N_PIXELS = 32
IMAGE_DIM = 1_024
SUMMARY_DIM = 6
N_STATES = 20
N_POSTERIOR = 2_000
RFF_DIM = 1_024
EPSILONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
METHODS = ("NPE", "NPE-MDS (RF)")
DATASET_BATCH = 8


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


def build_task() -> tuple[CryoEMTask, dict]:
    source_config_path = OFFICIAL / "configs" / "cyro_em_sim_config.json"
    config = json.loads(source_config_path.read_text(encoding="utf-8"))
    config["MODEL_FILE"] = str(
        (CRYOSBI / "tests" / "models" / "hsp90_models.pt").resolve()
    )
    resolved_path = ARTIFACT / "resolved_sim_config.json"
    resolved_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cfg = SimpleNamespace(
        sim_config_path=str(resolved_path),
        n_obs=N_OBS,
        device="cpu",
    )
    task = CryoEMTask(cfg)
    if task.num_models != N_STATES or task.num_pixels != N_PIXELS:
        raise AssertionError(
            f"unexpected HSP90 task: states={task.num_models}, pixels={task.num_pixels}"
        )
    return task, config


def simulate_dataset_batches(
    task: CryoEMTask,
    theta: torch.Tensor,
    seed: int,
    *,
    progress_label: str,
):
    """Yield deterministic full datasets without retaining the whole corpus."""
    seed_all(seed)
    for start in range(0, len(theta), DATASET_BATCH):
        values = theta[start : start + DATASET_BATCH, 0]
        indices = values.repeat_interleave(N_OBS).float().unsqueeze(-1)
        images = task.simulator.simulate(
            num_sim=len(indices),
            indices=indices,
            return_parameters=False,
            batch_size=32,
        )
        images = images.reshape(len(values), N_OBS, N_PIXELS, N_PIXELS)
        if start % 200 == 0:
            print(
                f"CLAIM6_SIM_PROGRESS label={progress_label} "
                f"datasets={start + len(values)}/{len(theta)}",
                flush=True,
            )
        yield start, images


def generate_training_corpus(
    task: CryoEMTask, temp_dir: Path
) -> tuple[torch.Tensor, torch.Tensor, Path, dict]:
    theta_gen = torch.Generator().manual_seed(SEED)
    theta = torch.randint(0, N_STATES, (N_TRAIN, 1), generator=theta_gen).float()
    summaries = torch.empty(N_TRAIN, SUMMARY_DIM)
    raw_path = temp_dir / "cryo_rff_train.npy"
    raw = np.lib.format.open_memmap(
        raw_path,
        mode="w+",
        dtype=np.float32,
        shape=(N_RFF_DATASETS, N_OBS, IMAGE_DIM),
    )
    started = time.perf_counter()
    for start, images in simulate_dataset_batches(
        task, theta, SEED + 1, progress_label="train"
    ):
        count = len(images)
        summaries[start : start + count] = task.compute_summary_statistics(images)
        if start < N_RFF_DATASETS:
            take = min(count, N_RFF_DATASETS - start)
            raw[start : start + take] = images[:take].reshape(take, N_OBS, IMAGE_DIM)
    raw.flush()
    receipt = {
        "datasets": N_TRAIN,
        "rff_datasets_materialized": N_RFF_DATASETS,
        "raw_bytes": raw_path.stat().st_size,
        "raw_sha256": sha256(raw_path),
        "generation_seconds": time.perf_counter() - started,
        "state_counts": {
            str(state): int((theta[:, 0] == state).sum()) for state in range(N_STATES)
        },
    }
    return theta, summaries, raw_path, receipt


def transform_dataset_blocks(
    rff: RBFSampler, raw: np.memmap, indices: np.ndarray
) -> np.ndarray:
    output = np.empty((len(indices), RFF_DIM), dtype=np.float32)
    for start in range(0, len(indices), DATASET_BATCH):
        selected = indices[start : start + DATASET_BATCH]
        block = np.asarray(raw[selected], dtype=np.float32)
        transformed = rff.transform(block.reshape(-1, IMAGE_DIM))
        output[start : start + len(selected)] = transformed.reshape(
            len(selected), N_OBS, RFF_DIM
        ).mean(axis=1)
        if start % 200 == 0:
            print(
                f"CLAIM6_RFF_PROGRESS datasets={start + len(selected)}/{len(indices)}",
                flush=True,
            )
    return output


def fit_rff_adapter(
    raw_path: Path,
    summaries: torch.Tensor,
    theta: torch.Tensor,
) -> tuple[RFFTTAAdapter, np.ndarray, dict]:
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
    raw = np.load(raw_path, mmap_mode="r")
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(N_RFF_DATASETS)
    n_calib = int(0.05 * N_RFF_DATASETS)
    calib_idx, train_idx = permutation[:n_calib], permutation[n_calib:]

    # Exact official median-heuristic selection over the post-split image pool.
    median_rng = np.random.default_rng(SEED)
    flat_positions = median_rng.choice(
        len(train_idx) * N_OBS, size=5_000, replace=False
    )
    median_datasets = train_idx[flat_positions // N_OBS]
    median_images = flat_positions % N_OBS
    subset = np.asarray(raw[median_datasets, median_images], dtype=np.float32)
    squared_norms = np.sum(subset**2, axis=1)
    squared = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * subset @ subset.T
    )
    median_squared_distance = float(
        np.median(squared[np.triu_indices(len(subset), k=1)])
    )
    adapter.gamma = 1.0 / (2.0 * median_squared_distance + 1e-8)
    adapter.rff = RBFSampler(
        gamma=adapter.gamma,
        n_components=RFF_DIM,
        random_state=SEED,
    )
    adapter.rff.fit(subset)

    transform_started = time.perf_counter()
    z_train = transform_dataset_blocks(adapter.rff, raw, train_idx)
    z_calib = transform_dataset_blocks(adapter.rff, raw, calib_idx)
    transform_seconds = time.perf_counter() - transform_started

    s_train = summaries[train_idx]
    s_calib = summaries[calib_idx]
    adapter.s_mean = s_train.mean(0)
    adapter.s_std = s_train.std(0) + 1e-8
    s_norm = (s_train - adapter.s_mean) / adapter.s_std
    adapter.regressor = adapter._build_regressor(SUMMARY_DIM, RFF_DIM)
    regressor_started = time.perf_counter()
    adapter.regressor_losses = adapter._train_regressor(
        s_norm, torch.as_tensor(z_train, dtype=torch.float32)
    )
    regressor_seconds = time.perf_counter() - regressor_started
    adapter.s_mean = adapter.s_mean.to(adapter.device)
    adapter.s_std = adapter.s_std.to(adapter.device)

    with torch.no_grad():
        calib_norm = (
            s_calib.to(adapter.device) - adapter.s_mean
        ) / adapter.s_std
        predicted = adapter.regressor(calib_norm).cpu().numpy()
    calib_losses = np.sum((predicted - z_calib) ** 2, axis=1)
    adapter.calibrated_tau = float(np.quantile(calib_losses, 0.95))
    adapter.config.tau = adapter.calibrated_tau

    z_all = np.empty((N_RFF_DATASETS, RFF_DIM), dtype=np.float32)
    z_all[train_idx] = z_train
    z_all[calib_idx] = z_calib
    state_embeddings = np.stack(
        [
            z_all[theta[:N_RFF_DATASETS, 0].numpy() == state].mean(axis=0)
            for state in range(N_STATES)
        ]
    )
    receipt = {
        "rff_dimension": RFF_DIM,
        "fit_datasets": N_RFF_DATASETS,
        "train_datasets_after_calibration": len(train_idx),
        "calibration_datasets": len(calib_idx),
        "gamma": float(adapter.gamma),
        "median_squared_distance": median_squared_distance,
        "calibrated_tau": adapter.calibrated_tau,
        "calibration_loss_mean": float(calib_losses.mean()),
        "calibration_loss_std": float(calib_losses.std(ddof=1)),
        "regressor_final_mse": float(adapter.regressor_losses[-1]),
        "rff_transform_seconds": transform_seconds,
        "regressor_fit_seconds": regressor_seconds,
        "stream_block_datasets": DATASET_BATCH,
    }
    return adapter, state_embeddings, receipt


def train_npe(
    theta: torch.Tensor, summaries: torch.Tensor
) -> tuple[torch.nn.Module, dict, float]:
    seed_all(SEED)
    model = build_npe_model(
        theta_sample=theta,
        x_sample=summaries,
        embedding_type="none",
    )
    config = NPE_TrainConfig(
        lr=5e-4,
        batch_size=256,
        val_frac=0.1,
        stop_after_epochs=20,
        max_epochs=500,
    )
    started = time.perf_counter()
    model, history = train_NPE_estimator(
        model, theta, summaries, config=config, device="cpu", seed=SEED
    )
    model.eval()
    return model, history, time.perf_counter() - started


def deterministic_simulator_audit(task: CryoEMTask) -> dict:
    theta = torch.tensor([[7.0]])
    first = next(
        simulate_dataset_batches(task, theta, SEED + 90_000, progress_label="audit_a")
    )[1]
    second = next(
        simulate_dataset_batches(task, theta, SEED + 90_000, progress_label="audit_b")
    )[1]
    third = next(
        simulate_dataset_batches(task, theta, SEED + 90_001, progress_label="audit_c")
    )[1]
    return {
        "same_seed_bit_identical": bool(torch.equal(first, second)),
        "different_seed_changes_images": bool(not torch.equal(first, third)),
        "same_seed_sha256": hashlib.sha256(first.numpy().tobytes()).hexdigest(),
        "different_seed_sha256": hashlib.sha256(third.numpy().tobytes()).hexdigest(),
    }


def generate_clean_test(task: CryoEMTask) -> tuple[torch.Tensor, torch.Tensor]:
    theta_gen = torch.Generator().manual_seed(SEED + 1_000)
    theta = torch.randint(0, N_STATES, (N_TEST, 1), generator=theta_gen).float()
    batches = [
        images
        for _, images in simulate_dataset_batches(
            task, theta, SEED + 1_001, progress_label="test_clean"
        )
    ]
    return theta, torch.cat(batches, dim=0)


def contaminate(
    task: CryoEMTask, clean: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    observed = clean.clone()
    mask = torch.zeros(N_TEST, N_OBS, dtype=torch.bool)
    if epsilon == 0.0:
        return observed, mask
    k = int(epsilon * N_OBS)
    mask_gen = torch.Generator().manual_seed(SEED + 1_000 + 10_000)
    for sample_index in range(N_TEST):
        permutation = torch.randperm(N_OBS, generator=mask_gen)
        mask[sample_index, permutation[:k]] = True
        noise = task._generate_pure_noise(
            k, seed=SEED + 1_000 + 20_000 + sample_index
        )
        observed[sample_index, mask[sample_index]] = noise
    return observed, mask


def posterior_rmse(samples: torch.Tensor, theta: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((samples.mean(0) - theta) ** 2)))


def predictive_rff_mmd(
    samples: torch.Tensor,
    clean_images: torch.Tensor,
    state_embeddings: np.ndarray,
    rff: RBFSampler,
) -> float:
    states = (
        samples[:, 0].round().clamp(0, N_STATES - 1).to(torch.int64).numpy()
    )
    weights = np.bincount(states, minlength=N_STATES).astype(np.float64)
    weights /= weights.sum()
    predicted_embedding = weights @ state_embeddings
    clean_embedding = rff.transform(
        clean_images.reshape(N_OBS, IMAGE_DIM).numpy()
    ).mean(axis=0)
    return float(np.sum((predicted_embedding - clean_embedding) ** 2))


def bootstrap_ci(
    delta: np.ndarray, seed: int, replicates: int = 10_000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for start in range(0, replicates, 1_000):
        size = min(1_000, replicates - start)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        means[start : start + size] = delta[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for epsilon_index, epsilon in enumerate(EPSILONS):
        selected = {
            method: [
                row
                for row in rows
                if row["epsilon"] == epsilon and row["method"] == method
            ]
            for method in METHODS
        }
        npe_rmse = np.array([row["rmse"] for row in selected["NPE"]])
        mds_rmse = np.array([row["rmse"] for row in selected["NPE-MDS (RF)"]])
        npe_pred = np.array(
            [row["predictive_rff_mmd"] for row in selected["NPE"]]
        )
        mds_pred = np.array(
            [row["predictive_rff_mmd"] for row in selected["NPE-MDS (RF)"]]
        )
        rmse_low, rmse_high = bootstrap_ci(
            npe_rmse - mds_rmse, SEED + 40_000 + epsilon_index
        )
        pred_low, pred_high = bootstrap_ci(
            npe_pred - mds_pred, SEED + 50_000 + epsilon_index
        )
        for method in METHODS:
            rmses = np.array([row["rmse"] for row in selected[method]])
            preds = np.array(
                [row["predictive_rff_mmd"] for row in selected[method]]
            )
            output.append(
                {
                    "epsilon": epsilon,
                    "method": method,
                    "trials": len(rmses),
                    "rmse_mean": float(rmses.mean()),
                    "rmse_std": float(rmses.std(ddof=1)),
                    "predictive_rff_mmd_mean": float(preds.mean()),
                    "predictive_rff_mmd_std": float(preds.std(ddof=1)),
                    "npe_minus_mds_rmse_mean": float(
                        (npe_rmse - mds_rmse).mean()
                    ),
                    "npe_minus_mds_rmse_ci95_low": rmse_low,
                    "npe_minus_mds_rmse_ci95_high": rmse_high,
                    "npe_minus_mds_predictive_mean": float(
                        (npe_pred - mds_pred).mean()
                    ),
                    "npe_minus_mds_predictive_ci95_low": pred_low,
                    "npe_minus_mds_predictive_ci95_high": pred_high,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(REQUESTED_THREADS)
    torch.set_num_interop_threads(1)
    seed_all(SEED)
    started = time.perf_counter()
    task, resolved_config = build_task()
    simulator_audit = deterministic_simulator_audit(task)
    if not all(
        [
            simulator_audit["same_seed_bit_identical"],
            simulator_audit["different_seed_changes_images"],
        ]
    ):
        raise AssertionError("deterministic simulator audit failed")

    with tempfile.TemporaryDirectory(prefix="mds-cryo-") as temp:
        theta_train, summaries, raw_path, generation_receipt = (
            generate_training_corpus(task, Path(temp))
        )
        model, history, npe_seconds = train_npe(theta_train, summaries)
        model_hash_before = state_hash(model)
        adapter, state_embeddings, rff_receipt = fit_rff_adapter(
            raw_path, summaries, theta_train
        )

        posterior = TTAPosterior(
            model, adapter, device="cpu", bypass_embedding=True
        )
        theta_test, clean = generate_clean_test(task)
        rows: list[dict] = []
        for epsilon_index, epsilon in enumerate(EPSILONS):
            observed, mask = contaminate(task, clean, epsilon)
            observed_summaries = task.compute_summary_statistics(observed)
            for sample_index in range(N_TEST):
                theta_true = theta_test[sample_index]
                x_obs = observed[sample_index]
                s_obs = observed_summaries[sample_index]
                seed_all(
                    SEED + 300_000 + 10_000 * epsilon_index + 10 * sample_index
                )
                npe_samples = sample_npe_posterior(
                    model, s_obs, N_POSTERIOR, "cpu"
                )
                seed_all(
                    SEED
                    + 300_001
                    + 10_000 * epsilon_index
                    + 10 * sample_index
                )
                adapt_started = time.perf_counter()
                mds_samples, adapt_info = posterior.sample(
                    x_obs.reshape(N_OBS, IMAGE_DIM).numpy(),
                    n_samples=N_POSTERIOR,
                    adapt=True,
                    return_info=True,
                    s_obs=s_obs,
                )
                adapt_ms = 1_000 * (time.perf_counter() - adapt_started)
                for method, samples in (
                    ("NPE", npe_samples),
                    ("NPE-MDS (RF)", mds_samples),
                ):
                    row = {
                        "epsilon": epsilon,
                        "sample_index": sample_index,
                        "method": method,
                        "true_state": int(theta_true.item()),
                        "posterior_mean": float(samples.mean()),
                        "rmse": posterior_rmse(samples, theta_true),
                        "predictive_rff_mmd": predictive_rff_mmd(
                            samples,
                            clean[sample_index],
                            state_embeddings,
                            adapter.rff,
                        ),
                        "actual_contamination_fraction": float(
                            mask[sample_index].float().mean()
                        ),
                        "mds_gate_passed": (
                            bool(adapt_info["gate_passed"])
                            if method == "NPE-MDS (RF)"
                            else ""
                        ),
                        "mds_lbfgs_evaluations": (
                            int(adapt_info["n_steps"])
                            if method == "NPE-MDS (RF)"
                            else ""
                        ),
                        "mds_adapt_and_sample_ms": (
                            adapt_ms if method == "NPE-MDS (RF)" else ""
                        ),
                    }
                    rows.append(row)
                    print(
                        "CLAIM6_TRIAL " + json.dumps(row, sort_keys=True),
                        flush=True,
                    )

        raw_output = ARTIFACT / "raw_trials.csv"
        write_csv(raw_output, rows)
        aggregate_rows = aggregate(rows)
        write_csv(ARTIFACT / "aggregate.csv", aggregate_rows)
        model_hash_after = state_hash(model)
        summary = {
            "claim": "Claim 6 full 1024D HSP90 Cryo-EM reproduction",
            "paper_scope": {
                "states": N_STATES,
                "training_datasets": N_TRAIN,
                "rff_fit_datasets": N_RFF_DATASETS,
                "test_datasets": N_TEST,
                "images_per_dataset": N_OBS,
                "image_shape": [N_PIXELS, N_PIXELS],
                "image_dimension": IMAGE_DIM,
                "summary_dimension": SUMMARY_DIM,
                "posterior_samples": N_POSTERIOR,
                "epsilons": EPSILONS,
                "contamination": "per-image normalized pure Gaussian noise",
                "rff_dimension_task_config": RFF_DIM,
            },
            "model_hashes": {
                "npe_before_adaptation": model_hash_before,
                "npe_after_adaptation": model_hash_after,
            },
            "training": {
                "seconds": npe_seconds,
                "best_epoch": history["best_epoch"],
                "best_val_loss": history["best_val_loss"],
            },
            "generation": generation_receipt,
            "rff": rff_receipt,
            "simulator_audit": simulator_audit,
            "aggregate": aggregate_rows,
            "raw_sha256": sha256(raw_output),
            "source": {
                "paper_tex_sha256": sha256(
                    ROOT / "source" / "arxiv" / "arxiv_main.tex"
                ),
                "official_repo_commit": "45158124f0cbdc2f6c1ac602c9fc5501dce20af3",
                "cryosbi_commit": "8e5832ecda626e9ab58d18cb215b5db6789533ee",
                "task_config_sha256": sha256(
                    OFFICIAL / "configs" / "cyro_em_config.yaml"
                ),
                "sim_config_sha256": sha256(
                    OFFICIAL / "configs" / "cyro_em_sim_config.json"
                ),
                "task_module_sha256": sha256(
                    OFFICIAL / "src" / "tt_sbi" / "tasks" / "cyro_em.py"
                ),
                "hsp90_models_sha256": sha256(
                    CRYOSBI / "tests" / "models" / "hsp90_models.pt"
                ),
            },
            "resolved_simulator_config": resolved_config,
            "reproducibility": {
                "git_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "fixed_command": "git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py",
                "uv_lock_sha256": sha256(ROOT / "uv.lock"),
                "seed": SEED,
            },
            "compute": {
                "estimated_required_cores": 32,
                "selected_backend": "hf",
                "selected_flavor": "cpu-upgrade",
                "actual_cpu_allocation_logical": os.cpu_count(),
                "configured_torch_threads": REQUESTED_THREADS,
                "gpu_used": False,
                "platform": platform.platform(),
                "runtime_seconds": time.perf_counter() - started,
            },
            "metric_boundary": {
                "primary": "posterior RMSE against true state, paper exact",
                "secondary": "squared 1024-RFF posterior-predictive mean-embedding distance",
                "deviation": "The paper plots exact five-bandwidth predictive MMD. Full exact 15,000-image pairwise kernels per method/test are not CPU-feasible; the secondary metric uses the same full-data RFF kernel approximation as MDS and is not represented as exact MMD.",
            },
        }
        (ARTIFACT / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("CLAIM6_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
