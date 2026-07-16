# Claim 1 - Plug-in MMD summary independent of NPE


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d0c1f46308b1", "created_at": "2026-07-16T15:51:57+00:00", "title": "VERIFIED - two trained frozen neural posteriors"}
-->
## Exact catalog claim

> Minimum-distance summaries provide a plug-in robust neural posterior estimation method that adapts test-time summaries independently of pretrained NPE using maximum mean discrepancy.

## Verdict: VERIFIED

The primary source defines `s*(Q) = argmin_s D(P_x|s,Q)` at `arxiv_main.tex:236`, specializes `D` to MMD/RFF at lines 283-285, and returns `q_psi(theta | s*)` in Algorithm 1 beginning at line 289. The pretrained NPE is queried, not retrained.

The decisive test trains and checkpoints two genuine conditional-density neural posterior estimators: a 4,418-parameter Gaussian NPE on 20,000 clean simulations and a 17,540-parameter OUP NPE on 10,000 clean simulations. During testing, every NPE parameter is frozen. Tensor-state SHA-256 is asserted identical before and after every adaptation: Gaussian `38161a0ab9d3dfc5b0893964094d4c062298d16d41a62890b03e8ce657fba5aa`; OUP `4e856211d806c14961078b3b8e0469f5a1db151d9afebdb785c1b4b91407ff50`.

For each held-out dataset the exact same frozen neural posterior is queried twice: once with the ordinary observed summary and once with the official-RFF MDS summary. MDS never receives, reads, optimizes, or mutates NPE weights. The Gaussian test contains 300 paired queries; the OUP test contains 250. This verifies the claimed post-hoc plug-in separation directly, not through a conjugate-posterior proxy.

The earlier exact conjugate Gaussian analysis is retained only as an independently checkable reference for the summary optimizer and theorem special cases; it is no longer the sole evidence for this claim.

**Source integrity:** PDF SHA-256 `1fc774ab166496d0861720b14212204c46cf920dc22c5df006d1d48f5eba01f0`; arXiv TeX SHA-256 `ff81fd973e3bcba86fb23e9a0c102ec88e240f62361315c7875de54e29ea4fd2`; source/code anchors are exported in `source_and_code_audit.json`.
