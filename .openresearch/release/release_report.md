# Final release report

- Previous live judged score: `7/12`
- Conservative projected score range after the proposed change: `9–10/12`
- Best-supported possible new score: `10/12` — forecast, not a judge result

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
| --- | ---: | ---: | --- | --- | --- |
| 1 | 2/2 | 2/2 | HIGH | VERIFIED | Frozen NPE hashes and 550 paired adaptations pass cumulatively. |
| 2 | 2/2 | 2/2 | HIGH | VERIFIED | Official 512-RFF/MSE/L-BFGS path and controls pass cumulatively. |
| 3 | 1/2 | 2/2 | HIGH | FALSIFIED | Exact-assumption counterexample; interpretation of the printed Hessian is the residual judge risk. |
| 4 | 1/2 | 2/2 | HIGH | FALSIFIED | Exact conditional counterexample; omitted topology is the residual judge risk. |
| 5 | 1/2 | 2/2 | HIGH | FALSIFIED | Full comparator run falsifies only the imported stronger all-comparator gloss. |
| 6 | 0/2 | 0/2 | LOW | BLOCKED | Four routes complete; exact author checkpoint/data realization remains unavailable. |

The current live total remains `7/12`. Claims 3–5 changed scientifically
since the previous verdict; Claim 6 remains BLOCKED. The scientific winner is
`orx/integrated-claims-1-6-cumulative-evidence` at
`41ec2e32d267e6944c04cc766c495f59a3071fdb`, with cumulative run
`fd04a075-47dd-487c-a947-c6972227a67b`.

All 14 Hugging Face jobs through the scientific winner used `cpu-upgrade` and
no GPU. Total scheduler time was 16,418 seconds (4.5606 job-hours); at the
2026-07-26 listed rate of $0.03/hour, cost was $0.1368. The final cumulative
run took 1,709.112072 seconds, exposed 64 logical CPUs, and had an estimated
six useful cores.

The protected judged tree has 19 files. It is a subset of the candidate:
16 hashes are identical, and only the existing `README.md`, `logbook.json`,
and `pages/index.md` are additively updated. The canonical traversal and
secret/link checks pass. Exact upload paths and hashes are in
`upload_allowlist.txt` and `candidate_space.sha256`; the command history is in
`command_log.md`.

The exact publication action is one text-only commit to the existing
`DineshAI/lq8fNVME8v` Space, parented to
`a9e77b682e084d5174725eef5adfb172cb184e67`. No second Space or logbook is
created. The published revision will then be freshly downloaded, hash-checked,
and traversed again before the same reader-facing revision is pushed to
GitHub `master`.
