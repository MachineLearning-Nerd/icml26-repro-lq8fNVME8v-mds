# Current verification — Theorem 4.2

**Verdict: FALSIFIED.** This exact conditional counterexample supersedes the
historical finite Gaussian consistency check. It targets the theorem's printed
scope, not the Gaussian kernel used in the paper's experiments.

## Exact claim contract and source

Theorem 4.2 says that, under Appendix A.3's four assumptions and correct
specification, weak consistency of the original-summary posterior implies weak
consistency of the MDS-summary posterior.

Source: arXiv 2602.09161, TeX lines 368–374; assumptions lines 686–714;
metrization step lines 706–789; HTML anchor `#S4.Thmtheorem2`. Retrieved
2026-07-25 with an explicit User-Agent. TeX SHA-256:
`ff81fd973e3bcba86fb23e9a0c102ec88e240f62361315c7875de54e29ea4fd2`.

Falsification requires a correctly specified exact-conditional model
satisfying every printed assumption where the premise holds and the
conclusion does not.

## Construction and assumption audit

Let the parameter and data space be the closed, locally compact subset
\[
X=\{0,\sqrt2\}\cup\{n,1/n:n\ge2\}\subset\mathbb R.
\]
Use prior masses \(p(0)=p(\sqrt2)=1/4\) and
\(p(n)=p(1/n)=2^{-(n+1)}\), deterministic simulator \(x_i=\theta\), and
true parameter \(\theta_0=0\). The summary space is the fixed set
\(\{A,B,C\}\): at sample size \(N\), \(A\) means
\(x_1\in\{0,1/N\}\), \(B\) means \(x_1=N\), and \(C\) means otherwise.

The model is correctly specified, its exact regular conditionals exist, the
identity simulator is continuous, mixtures are strongly identifiable, and
the finite summary-space argmin exists. Under the observed all-zero data,
the original posterior is supported on \(\{0,1/N\}\), with
\(W_1(T_{A,N},\delta_0)=w_N/N\to0\): the theorem's premise holds.

Use
\[
\kappa(x,y)=1+g(x)e^{-(x-y)^2}g(y),\quad g(x)=x^2e^{-x^2}.
\]
It is bounded and continuous. It is ISPD because
\(\mathcal E_\kappa(\mu)=\mu(X)^2+\mathcal E_{\rm RBF}(g\mu)\);
zero energy forces \(\mu(X)=0\) and \(g\mu=0\). Since \(g\) vanishes only
at zero, \(\mu\) is supported at zero, and its zero total mass forces
\(\mu=0\).

## Exact contradiction and raw results

Nevertheless,
\(\mathrm{MMD}_\kappa(\delta_N,\delta_0)=N^2e^{-N^2}\to0\).
For every \(N=3,\ldots,30\), the exact MDS objective strictly selects summary
\(B\), whose posterior is \(\delta_N\).

| N | original W₁ to δ₀ | MMD(original) | MMD(δₙ) | h(N) |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 6.667e-2 | 1.989e-2 | 1.111e-3 | 0.900000 |
| 10 | 1.949e-4 | 1.930e-5 | 3.720e-42 | 0.990099 |
| 20 | 9.537e-8 | 4.756e-9 | 7.661e-172 | 0.997506 |
| 30 | 6.209e-11 | 2.067e-12 | underflow below 1e-390 | 0.998890 |

The bounded continuous witness \(h(x)=x^2/(1+x^2)\) has
\(E_{\delta_N}h\to1\), while \(h(0)=0\). Thus \(\delta_N\) does not
converge weakly to \(\delta_0\), directly contradicting the conclusion.

Download:
[all 28 raw rows](formal_results.csv),
[formal result and symbolic certificates](formal_result.json),
[independent checker output](independent_checker_output.json),
[negative-control output](negative_control_output.json), and
[compute receipt](runtime.json).

Executable fail-closed verifier:
[`python verify.py`](verify.py). It exits nonzero if the sweep, strict argmin,
premise, weak-convergence witness, checker, control, CPU receipt, or cumulative
regression gate fails.

## Diagnosis, independent check, and control

The cited Simon-Gabriel et al. result (arXiv:2006.09268) requires the
locally compact topology plus \(H_k\subset C_0\) for MMD to metrize weak
convergence. It also documents that bounded continuous ISPD alone is
insufficient. Appendix A.3 omits the \(C_0\) condition but invokes the
metrization implication. The independent algebraic checker reconstructed all
objectives and the ISPD certificate; all 7 checks passed.

With a Gaussian \(C_0\) kernel, the escaping summary has MMD
`1.4142135624` at \(N=30\), versus `8.778e-11` for the original summary.
The falsification program exits `1` and reports `NOT_FALSIFIED`, as intended.

## Command, environment, and limitations

Fixed command:

```text
git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py
```

Formal run `93bf6002-6dae-414e-a678-b2b39d42f138`, Git
`21cbbcff42d077e5508b77624b9d7691b2636b3d`, pinned Python 3.12 / `uv.lock`,
Hugging Face `cpu-upgrade`, 64 allocated logical CPUs, no GPU. The theorem
route used one core and 0.181511 seconds; the inherited cumulative command
took 831.358078 seconds and passed 16/16 regressions.

Adding the omitted \(H_k\subset C_0\) and associated topological conditions
blocks this counterexample and repairs this proof step. No learned
approximation, finite-sample neural behavior, or empirical Gaussian claim is
being assessed here.
