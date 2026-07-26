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
