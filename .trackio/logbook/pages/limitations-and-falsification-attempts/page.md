# Limitations and falsification attempts


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_4331f2b79cbe", "created_at": "2026-07-16T15:52:05+00:00", "title": "What this result does and does not establish"}
-->
# Limitations and negative evidence

- **Severe contamination:** at Gaussian eps=0.5, the default single-start RFF optimizer improves actual neural-NPE RMSE only 6.05%, wins 37/50, and diverges from the global exact-MMD mode. The retained conjugate reference gives the same 37/50 wins and 6.01% reduction. This reproduces the paper's warning that MDS can degrade at severe contamination.
- **Task scope:** Gaussian and OUP were rerun with trained frozen neural posterior networks. SIR, cryo-EM, NPE-PFN, robust-NPE comparator training, and the full paper figure suite were not rerun.
- **OUP scale:** the OUP test retains the official 25 timesteps, 100 trajectories, 512 RFFs and 50 held-out tests per level, but is a reduced CPU reproduction rather than the paper's entire benchmark sweep.
- **Timing scope:** reported milliseconds measure summary adaptation; neural-posterior query time is not included. This is not a normalizing-flow end-to-end latency benchmark.
- **Official package:** the public repo's top-level `tt_sbi.tta` import reaches undeclared `sbibm`; the isolated official `rff.py` is runnable and hash-pinned. The paper says 512 RFFs while the current dataclass defaults to 256; this run explicitly uses 512.
- **Theory scope:** Theorem 4.1 is infinitesimal/local. Theorem 4.2 assumes exact conditionals and strong identifiability. Neither theorem promises arbitrary-contamination robustness for an approximate learned decoder/NPE.
- **Terminology:** "model-free" means no likelihood or explicit error/contamination model at test time. The method still requires clean offline simulator/summary pairs to learn the conditional mean embedding.

These failures and caveats are included in the scored evidence because they define precisely where the four catalog claims are supported.
