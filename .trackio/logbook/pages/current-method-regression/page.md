# Current cumulative regression — Claims 1 and 2

**Verdicts: Claim 1 VERIFIED; Claim 2 VERIFIED.** This page is a compact,
downloadable entrypoint for the two previously full-credit methodological
claims. It does not replace or alter their detailed judged pages; it makes
their current cumulative code and receipt directly discoverable.

## Exact claims

1. MDS adapts the test-time query summary
   \(s^*=\arg\min_s D(P_{x\mid s},Q)\) while keeping the pretrained neural
   posterior fixed, preserving amortization (Section 3.1).
2. MDS uses a Gaussian-kernel MMD objective approximated with random Fourier
   features, an offline MSE-trained decoder mean-embedding network, and
   test-time L-BFGS (Section 3.2 / Algorithm 1).

Source TeX SHA-256:
`ff81fd973e3bcba86fb23e9a0c102ec88e240f62361315c7875de54e29ea4fd2`.
The exact source anchors and detailed raw blocks remain on the
[Claim 1 judged page](#/claim-1-plug-in-mmd-summary-independent-of-npe) and
[Claim 2 judged page](#/claim-2-rff-efficiency-and-lightweight-adaptation).

## Current accepted evidence

| Mechanism | Frozen NPE parameters | Paired queries | Tensor hash before/after |
| --- | ---: | ---: | --- |
| Gaussian | 4,418 | 300 | `38161a0a…fba5aa` identical |
| OUP | 17,540 | 250 | `4e856211…07ff50` identical |

The official `rff.py` path contains and executes all seven required
mechanisms: `RBFSampler`, empirical embeddings, the conditional-embedding
regressor, MSE fit, CPU support, calibrated gate, and L-BFGS with
strong-Wolfe. Paper-scale 512 RFFs and the 2×256 regressor are explicit.
Gaussian offline fit took `5.702 s` with final embedding MSE `2.530e-06`;
median test adaptation was `7.186 ms` and p95 `11.999 ms`. OUP fit took
`1.687 s`; nonzero-contamination median adaptation was `137.877 ms`.

Download the
[machine-readable accepted-metrics receipt](accepted_metrics.json) and
[current cumulative compute receipt](cumulative_runtime.json).
Executable source:
[neural NPE experiment](neural_npe_upgrade.py),
[integration and independent checker](integrate_neural_upgrade.py),
[baseline mechanism verifier](test_reproduction.py), and
[exact/RFF control implementation](reproduce_mds.py).
The compact current receipts are the
[Claim 1 checker](claim1_checker_output.json),
[Claim 1 expected-failure control](claim1_negative_control_output.json),
[Claim 1 fail-closed verifier](verify_claim1.py),
[Claim 2 checker](claim2_checker_output.json),
[Claim 2 expected-failure control](claim2_negative_control_output.json), and
[Claim 2 fail-closed verifier](verify_claim2.py).

`test_reproduction.py` and the integration script exit nonzero when a frozen
hash, architecture, paired-query count, official mechanism, approximation,
or severe-contamination control fails. Both compact verifier controls were
also executed on corrupted receipts and exited `1` as intended. The important
scientific negative boundary is retained: at epsilon 0.5, the single-start
optimizer can select the wrong mode and the RFF/exact-summary gap grows to
`3.818`; this is reported, not hidden.

## Fixed command and latest cumulative run

```text
git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py
```

The latest cumulative scientific run was
`fd04a075-47dd-487c-a947-c6972227a67b` at Git
`41ec2e32d267e6944c04cc766c495f59a3071fdb`, Python 3.12 / exact
`uv.lock`, Hugging Face `cpu-upgrade`, 64 exposed logical CPUs, estimated six
useful cores, no GPU, `1709.112072` seconds. It reran Claims 1 and 2 before
every current Claim 3–6 verifier.
