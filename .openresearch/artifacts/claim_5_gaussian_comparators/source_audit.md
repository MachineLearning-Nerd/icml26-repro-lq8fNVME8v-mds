# Claim 5 source audit

Retrieval date: 2026-07-25

## Primary source

- ar5iv URL: `https://ar5iv.labs.arxiv.org/html/2602.09161`
- ar5iv HTML SHA-256: `f4d5f3cb369729e3d083fb09260d64e53127868dd25bae01be235f6a2bea4`
- arXiv e-print URL: `https://export.arxiv.org/e-print/2602.09161`
- e-print archive SHA-256: `74473bd7156ee81883e35dff79d6948657752ffae47f79ca9ed0a85a1d9ded3e`
- extracted TeX SHA-256: `ff81fd973e3bcba86fb23e9a0c102ec88e240f62361315c7875de54e29ea4fd2`
- Figure 2 PDF (`fig-gaussian.pdf`) SHA-256:
  `bb45dee1f24eff0d0aa4f3a061aa3bccd4ae3e293b18ad9c24be86ff39a44039`

## Anchors and quantifiers

- Main Gaussian statement: TeX lines 399–404, section
  `\label{exp:gaussian}` and figure `\label{fig:main-3}`.
- General experiment domain: TeX lines 387–391: Huber contamination,
  epsilon in 0.0 through 0.5, and 100 observations.
- Comparator definitions: TeX lines 393–397.
- RFF implementation: TeX lines 938–942: 512 features, median bandwidth,
  two 256-unit layers, and L-BFGS.
- Comparator details: TeX lines 946–952: NNPE rho=1, sigma=0.01, tau=0.2;
  OC-SVM median heuristic, 5% false-positive calibration, removal and
  resampling.
- Replication/metric details: TeX lines 954–958: 100 test datasets and MMD
  against the analytic reference posterior.
- Gaussian domain: TeX lines 978–986: 50,000 training datasets, 100 test
  datasets, dimension 2, 100 observations, contamination epsilon in
  {0.1,0.2,0.3,0.5}, and shift 3.

## Exact-language audit

The paper says MDS improves robustness compared with benchmarks “in general”
and can degrade under a severe contamination proportion. It does not state
that MDS wins against every comparator at every tested epsilon. The imported
judge claim is stronger: it names NPE, NNPE, and NPE-OR as methods MDS
outperforms “across a range.” The published right-hand Figure 2 visibly places
NPE-OR below both MDS curves at epsilon 0.3 and 0.5.

The machine contract therefore tests the stronger imported claim honestly.
Any falsification is explicitly scoped to that gloss and is not represented
as a contradiction of the paper's qualified statement.

## Code provenance

- Official repository commit:
  `45158124f0cbdc2f6c1ac602c9fc5501dce20af3`
- Gaussian config SHA-256:
  `40628b2ccc888780067bbef083e3da657aec235b69e983993f9ddba2398e9234`
- RFF module SHA-256:
  `0a7e8a2261c9ab9f8b327c01335dd7ff9cb7d5efd48abcf1980bf9c125d329b9`
- NPE module SHA-256:
  `650dd73d9a005aad438b8a9fbdfe5626cc24772afe7650b83be8506169a8c121`
- NNPE module SHA-256:
  `2e0cfef16ca8a2307c09a6d5b472f7c0bb7a1911e373c25115b580d38e718ba2`
- OC-SVM module SHA-256:
  `6cc06957d84623eb0ebfd87f9d92a622fb2ffd01653bd1f785423ae793023dce`
