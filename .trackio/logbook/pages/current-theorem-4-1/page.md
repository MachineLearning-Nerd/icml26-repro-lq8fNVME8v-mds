# Current verification — Theorem 4.1

**Verdict: FALSIFIED.** This page supersedes the finite Gaussian special-case
check in the historical logbook. The result is an analytic population
counterexample to the theorem exactly as printed; it is not a claim that the
paper's typical Gaussian experiments are non-robust.

## Exact claim contract and source

Theorem 4.1 states that, under Appendix A.1's ten assumptions, for **any**
target distribution \(Q\), the supremum over point-mass contaminations \(y\)
of the right derivative at \(\epsilon=0\) of
\(\mathrm{KL}\{P(\theta\mid s^*(Q)),P(\theta\mid
s^*(Q_{\epsilon,y}))\}\) is finite.

Source: arXiv 2602.09161, TeX lines 337–346; assumptions lines 497–557;
HTML anchor `#S4.Thmtheorem1`. The source was retrieved 2026-07-25 with an
explicit User-Agent. TeX SHA-256:
`ff81fd973e3bcba86fb23e9a0c102ec88e240f62361315c7875de54e29ea4fd2`.

Falsification requires one \(Q,y\) satisfying every printed assumption for
which the right KL difference quotient diverges. That is a direct test of the
universal quantifier, not finite corroboration.

## Construction and assumptions

Take \(S=[-1/2,1/2]\), decoder \(P_{x\mid s}=N(s,1)\), Gaussian kernel
\(\exp[-(x-y)^2/2]\), and
\(Q=(\delta_{-\sqrt2}+\delta_{\sqrt2})/2\). Contaminate at \(y=\sqrt2\).
Use the exact posterior density proportional to
\(\exp[-(\theta-s)^2]\) on \([-1,1]\).

All ten printed assumptions pass: the potential and its local gradient are
regular; \(S\) is convex; the kernel is bounded; the decoder has a Lebesgue
density that is \(C^2\) in \(s\) with uniform integrable derivatives; and the
paper-defined model-averaged Hessian is
\(M=2/(3\sqrt3)=0.3849001795>0\). The decisive mismatch is that the actual
arbitrary-\(Q\) objective Hessian is zero. Its unique minimum is quartically
flat.

The cited influence formula in Briol et al. (arXiv:1906.05944) is stated at a
model distribution. Applying its model-averaged Hessian to arbitrary \(Q\) is
the gap exercised here.

## Raw formal result

The exact asymptotics are
\[
s^*(Q_{\epsilon,\sqrt2})\sim(6\sqrt2\,\epsilon)^{1/3},\qquad
\mathrm{KL}/\epsilon\sim2.1109031920\,\epsilon^{-1/3}.
\]

| ε | adapted summary | posterior KL | KL / ε |
| ---: | ---: | ---: | ---: |
| 1e-3 | 0.203117900 | 2.0867738e-2 | 20.8677 |
| 1e-5 | 0.043934419 | 9.7927004e-4 | 97.9270 |
| 1e-7 | 0.009467127 | 4.5476900e-5 | 454.7690 |
| 1e-9 | 0.002039648 | 2.1109008e-6 | 2110.9008 |
| 1e-10 | 0.000946721 | 4.5478019e-7 | 4547.8019 |

Observed log-log slopes are `0.333156` for the summary and `-0.333822`
for KL/ε. The latter contradicts the claimed finite right derivative.

Download:
[raw JSON](formal_results.json),
[independent checker output](independent_checker_output.json),
[negative-control output](negative_control_output.json), and
[compute receipt](runtime.json).

Executable fail-closed verifier:
[`python verify.py`](verify.py). It exits nonzero if an assumption flag,
rate, monotonic divergence, checker, control, CPU receipt, or cumulative
regression gate is absent.

## Independent check and negative control

The independent checker reconstructed both constants and slopes without using
the verifier's pass decision: all 7 checks passed. At the smallest ε, the
scaled constants were `2.0396487192` versus analytic `2.0396489027`, and
`2.1109026661` versus analytic `2.1109031920`.

For the prespecified correctly specified control \(Q=N(0,1)\), the true target
Hessian equals `0.3849001795`, the summary moves at \(O(\epsilon)\), and
KL/ε tends to zero. The falsification program exited `1` as required and
reported `NOT_FALSIFIED`; a control that always passed could not satisfy this
gate.

## Command, environment, and limitations

Fixed command on every node:

```text
git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py
```

Formal run `e50ebf39-1199-4b76-9070-a7fa83e1b342`, Git
`e34e78537b699678bbb1fc02d03ce0141e273771`, pinned Python 3.12 / `uv.lock`,
Hugging Face `cpu-upgrade`, 64 allocated logical CPUs, no GPU. The theorem
route used one core and 1.171924 seconds; the inherited cumulative command
took 960.414060 seconds and passed 16/16 regressions.

The counterexample concerns the population theorem's printed universal scope.
It does not contradict the paper's empirical Gaussian/OUP results, nor assert
that typical correctly specified targets have unbounded influence.
