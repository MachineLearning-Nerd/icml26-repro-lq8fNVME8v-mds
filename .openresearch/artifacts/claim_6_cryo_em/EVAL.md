# Claim 6 evaluation

Status: BLOCKED

The full-scale code, exact claim contract, independent checker, and negative
control are prepared. No scientific run has occurred because Hugging Face Jobs
rejected submission with HTTP 402: pre-paid credit balance insufficient.

The result remains BLOCKED until a formal HF `cpu-upgrade` run completes and:

1. all inherited Claim 1–4 regression checks pass;
2. all 1,200 raw method-by-test rows are present;
3. the independent scale/integrity reconstruction passes;
4. the NPE tensor hash is identical before and after MDS;
5. the fail-closed exact-claim verifier exits zero; and
6. the disabled-adaptation control exits nonzero.

No forecast point is assigned while this page is blocked.
