# Claim 5 evaluation

Status: FALSIFIED

Formal cumulative run `fd04a075-47dd-487c-a947-c6972227a67b` completed at
Git commit `41ec2e32d267e6944c04cc766c495f59a3071fdb`. It used Hugging Face
`cpu-upgrade`, exposed 64 logical CPUs for an estimated six useful cores,
used no GPU, and completed the Claim 5 route in 1,655.65 seconds.

The full paper-scale design produced 2,000 raw method-by-test rows: 50,000
training datasets, 100 paired test datasets, 100 observations, two
dimensions, 2,000 posterior samples, epsilon in `{0,.1,.2,.3,.5}`, NPE,
NNPE, NPE-OR, and 512-RFF NPE-MDS. The raw CSV SHA-256 is
`3b813e9dc810b8b82abaac4631da3765e87951bcfbd1ad557d491697788c91bb`.
All 11 independent checks passed, both neural tensor-state hashes were
unchanged, and the disabled-adaptation control exited nonzero as intended.

The exact stronger imported claim is falsified. At epsilon 0.3, mean
posterior MMD was `2.0756` for MDS and `1.0626` for NPE-OR; the paired 95%
bootstrap interval for NPE-OR minus MDS was `[-1.3538,-0.6719]`, wholly
below zero. Thus MDS did not outperform every named comparator below 40%
contamination. MDS did outperform NPE and NNPE at epsilon 0.1–0.3, and its
mean MMD degraded to `4.7103` at epsilon 0.5 from `2.0756` at epsilon 0.3.

Scope boundary: this falsifies only the stronger imported all-comparator
gloss. It does not falsify the paper's qualified statement that MDS improves
robustness “in general.”

## Rejected setup attempts

- Run `021dcbe0-ddcc-47cc-9bde-f131d61fa75f` exited 127 before science because
  the default HF CPU image did not contain `uv`.
- Run `cb709f65-56ee-4945-82e7-f9b7a4ff5d9f` completed all cumulative Claim
  1–4 checks, then stopped before Claim 5 training because the authors'
  package initializer eagerly imported undeclared `sbibm`.

Neither attempt produced Claim 5 evidence. The current runner supersedes the
second setup path by bypassing only the unrelated eager initializer.
