# Current verification — Gaussian full comparator claim

**Verdict: FALSIFIED.** This page supersedes the Gaussian discussion in the
historical scored summary. The verdict applies to the stronger imported claim
that MDS outperforms all three named comparators below 40% contamination, not
to the paper's narrower statement that it improves robustness “in general.”

## Exact claim contract and source

The imported claim is: “On the Gaussian model, MDS outperforms NPE, NNPE, and
NPE-OR baselines across a range of contamination proportions, with
degradation only appearing at contamination levels above 40% (Figure 2).”

The paper's own Section 6.1 statement is qualified: “In general, our MDS
improves robustness compared to other benchmark methods, although for severe
proportion of contamination, MDS can degrade.” Source: arXiv 2602.09161,
Section 6.1 / Figure 2, TeX lines 400–412. The TeX was retrieved 2026-07-25
with an explicit User-Agent; SHA-256
`ff81fd973e3bcba86fb23e9a0c102ec88e240f62361315c7875de54e29ea4fd2`.

The preregistered falsification rule required MDS to beat NPE and NNPE at
epsilon 0.1–0.3, but required at least one paired 95% interval for NPE-OR
minus MDS to lie wholly below zero before 40% contamination. It also required
MDS at epsilon 0.5 to exceed 1.5 times its epsilon-0.3 value.

## Exact paper-scale experiment

| Item | Value |
| --- | --- |
| Training / paired tests | 50,000 / 100 |
| Observations / dimension | 100 / 2 |
| Posterior samples | 2,000 per method and test |
| Contamination | epsilon `{0,.1,.2,.3,.5}`, shift 3 |
| Methods | NPE, NNPE, NPE-OR, NPE-MDS |
| MDS | official implementation, 512 RFFs, L-BFGS |
| Metric | biased five-bandwidth MMD to analytic clean posterior |
| Uncertainty | paired bootstrap, 10,000 replicates |
| Seed | 42, with recorded deterministic offsets |

Mean posterior MMD:

| epsilon | NPE | NNPE | NPE-MDS | NPE-OR |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.1002 | 0.5855 | **0.0846** | 0.6063 |
| 0.1 | 1.5784 | 1.0635 | **0.4084** | 0.7594 |
| 0.2 | 2.9438 | 1.7338 | 1.0591 | **0.8810** |
| 0.3 | 4.0026 | 2.4915 | 2.0756 | **1.0626** |
| 0.5 | 5.1841 | 3.6236 | 4.7103 | **1.5444** |

At epsilon 0.3, NPE-OR minus MDS was `-1.0129`, with paired 95% bootstrap
interval `[-1.3538,-0.6719]`. This directly contradicts the all-comparator
gloss below 40%. MDS nevertheless beat NPE and NNPE at epsilon 0.1–0.3 and
degraded by a factor of `2.269` from epsilon 0.3 to 0.5.

## Raw evidence, executable verifier, and control

Download:
[2,000 raw rows](raw_trials.csv),
[aggregate table](aggregate.csv),
[formal summary](summary.json),
[independent checker](independent_checker_output.json),
[verifier output](verifier_output.json),
[negative-control output](negative_control_output.json), and
[compute receipt](runtime.json).

Executable source:
[experiment](gaussian_full_comparators.py),
[independent checker](check_gaussian_comparators.py), and
[fail-closed verifier](verify_gaussian_claim.py).
The verifier exits nonzero if scale, integrity, frozen-state, paired
contradiction, or degradation gates fail. The disabled-adaptation control
reconstructed MDS as ordinary NPE, obtained identically zero paired
differences, and exited `1` as required.

All 11 independent checks passed. The raw CSV SHA-256 is
`3b813e9dc810b8b82abaac4631da3765e87951bcfbd1ad557d491697788c91bb`.
The standard NPE tensor hash was unchanged at
`fb9cffe595631a6169c5e3548ebcd2b83c5cf3923d4fdbb35a764d57883251e6`;
the NNPE hash was unchanged at
`1f5d360747b6ece53bc97f8cd4a9ba1541cbb60a151b88a87fc3445d8e50bf9d`.

## Command, compute, and limitations

```text
git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py
```

Formal cumulative run `fd04a075-47dd-487c-a947-c6972227a67b`, Git
`41ec2e32d267e6944c04cc766c495f59a3071fdb`, Python 3.12, exact
`uv.lock`, Hugging Face `cpu-upgrade`, estimated six useful cores, 64
allocated logical CPUs, no GPU. Claim route runtime: `1655.648563` seconds;
cumulative runtime: `1709.112072` seconds.

This is one deterministic full-scale training realization. The formal
falsification does not require generalizing across seeds: the imported
all-comparator claim is contradicted by its paired NPE-OR result. It does not
falsify the paper's explicitly qualified “in general” wording.
