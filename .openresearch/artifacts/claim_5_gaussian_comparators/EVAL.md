# Claim 5 evaluation

Status: PENDING FORMAL RUN

The verdict will be exactly `FALSIFIED` or `BLOCKED` according to
`claim_contract.json`. A formal result is not accepted until:

1. the complete paper-scale run finishes on HF `cpu-upgrade`;
2. all cumulative Claim 1–4 checks pass;
3. `independent_checker_output.json` reports `PASS`;
4. `verifier_output.json` reports `FALSIFIED` and exits zero; and
5. the disabled-adaptation negative control exits nonzero.

No forecast point is assigned while this page says pending.

## Rejected setup attempts

- Run `021dcbe0-ddcc-47cc-9bde-f131d61fa75f` exited 127 before science because
  the default HF CPU image did not contain `uv`.
- Run `cb709f65-56ee-4945-82e7-f9b7a4ff5d9f` completed all cumulative Claim
  1–4 checks, then stopped before Claim 5 training because the authors'
  package initializer eagerly imported undeclared `sbibm`.

Neither attempt produced Claim 5 evidence. The current runner supersedes the
second setup path by bypassing only the unrelated eager initializer.
