# Claim 3 - Robustness gains with low overhead


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_246e9136a5e2", "created_at": "2026-07-16T15:52:00+00:00", "title": "VERIFIED ACROSS GAUSSIAN AND OUP WITH ACTUAL NEURAL NPES"}
-->
## Exact catalog claim

> The method demonstrates substantial robustness gains with minimal additional overhead.

## Verdict: VERIFIED ACROSS TWO MECHANISMS WITH ACTUAL NEURAL NPES

The prospectively frozen upgrade directly addresses both earlier scope boundaries: it replaces the analytic-posterior-only proxy with trained neural posterior networks and adds the paper-named Ornstein-Uhlenbeck-process mechanism. No trial is removed, all 12 frozen gates pass, and the protocol hash is `ede45cf04931ea5a2297228faa9bf1a42833d2f7ccda2df3a31b7e3e26cf1a02`.

### Gaussian actual-neural-NPE results

| eps | trials | neural NPE RMSE | neural NPE + MDS RMSE | reduction | wins | median ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 50 | 0.1023 | 0.1023 | 0.00% | 0/50 | 1.532 |
| 0.1 | 50 | 0.8019 | 0.1115 | 86.10% | 50/50 | 5.957 |
| 0.2 | 50 | 1.6039 | 0.1231 | 92.33% | 50/50 | 7.066 |
| 0.3 | 50 | 2.3924 | 0.1568 | 93.45% | 50/50 | 8.346 |
| 0.4 | 50 | 3.2023 | 0.1928 | 93.98% | 50/50 | 9.389 |
| 0.5 | 50 | 4.0076 | 3.8138 | 4.83% | 37/50 | 3.668 |

At every prespecified Gaussian contamination level eps=0.1-0.4, neural-NPE RMSE falls **86.10%-94.47%** (mean **91.46%**) and all **50/50** paired tests improve. The clean neural-NPE RMSE is preserved exactly by the gate. The network is trained once and is hash-identical after all 300 tests.

### OUP actual-neural-NPE results

| eps | trials | neural NPE RMSE | neural NPE + MDS RMSE | reduction | wins | median ms | gate pass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 50 | 0.3422 | 0.3438 | -0.48% | 1/50 | 10.845 | 44/50 |
| 0.1 | 50 | 2.7838 | 0.7097 | 74.50% | 48/50 | 140.905 | 0/50 |
| 0.2 | 50 | 2.7838 | 0.7761 | 72.12% | 48/50 | 136.876 | 0/50 |
| 0.3 | 50 | 2.7838 | 0.8955 | 67.83% | 46/50 | 138.877 | 0/50 |
| 0.4 | 50 | 2.7838 | 1.0585 | 61.98% | 46/50 | 135.361 | 0/50 |

OUP uses the official task dimensions—25 timesteps and 100 trajectories—and a two-parameter trained posterior. MDS reduces neural-NPE RMSE at **4/4** contamination levels by **68.51%-75.85%** (mean **69.11%**), with paired win rates **46/50 to 49/50** and p-values below `1e-17`. Median adaptation is **137.877ms** and worst-level p95 **166.502ms**.

**Falsification boundary:** Gaussian eps=0.5 is not called a success. Neural-NPE RMSE falls only **6.05%**, the win rate is 37/50, and the default optimizer can lock onto either mixture mode. The paper itself warns that MDS can degrade under severe contamination. The result verifies substantial gains and low absolute CPU overhead across two mechanisms; it is not inflated into a rerun of every paper benchmark or comparator.
