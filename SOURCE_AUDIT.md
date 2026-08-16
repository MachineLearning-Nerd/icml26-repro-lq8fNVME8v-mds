# Source and provenance audit

This repository separates primary paper material, official implementation
material, external dependencies, reproduction code, and generated evidence.
The separation prevents a reproduced result from being mistaken for an
official author release.

## Primary paper source

- Title: Minimum Distance Summaries for Robust Neural Posterior Estimation.
- Authors: Sherman Khoo, Dennis Prangle, Song Liu, and Mark Beaumont.
- arXiv: 2602.09161v2.
- OpenReview: lq8fNVME8v.
- PDF: paper/2602.09161.pdf.
- TeX: source/arxiv/arxiv_main.tex.
- PDF SHA-256:
  1fc774ab166496d0861720b14212204c46cf920dc22c5df006d1d48f5eba01f0.
- TeX SHA-256:
  ff81fd973e3bcba86fb23e9a0c102ec88e240f62361315c7875de54e29ea4fd2.

The source TeX contains the ICML author list and the theorem statements
audited by the formal routes. The PDF is retained so that displayed figures
and exact wording can be checked independently of the TeX source.

## Official implementation

- Repository:
  https://github.com/Shermjj/Minimum-Distance-Summaries.
- Pinned submodule commit:
  45158124f0cbdc2f6c1ac602c9fc5501dce20af3.
- Audited file:
  source/official-repo/src/tt_sbi/tta/rff.py.
- Audited file SHA-256:
  0a7e8a2261c9ab9f8b327c01335dd7ff9cb7d5efd48abcf1980bf9c125d329b9.

The working tree may show the official submodule as uninitialized after a
normal clone. Run git submodule update --init --recursive before executing
the producers. The commit pointer, rather than an unpinned moving default
branch, is the source identity used by the audit.

The package initializer imports ncpp, which imports utils.metrics and then
sbibm; sbibm is absent from the official pyproject dependency list. The
reproduction therefore loads the self-contained rff.py module directly. This
is a packaging workaround and is disclosed rather than silently altered.

The paper uses 512 RFFs, while the official repository default is 256. The
reproduction override to 512 is recorded in
outputs/full/source_and_code_audit.json and experiment_summary.json.

## External dependency

- Repository: https://github.com/flatironinstitute/cryoSBI.
- Pinned gitlink:
  8e5832ecda626e9ab58d18cb215b5db6789533ee.
- Role: Cryo-EM simulation and supporting route code.

The external dependency is not treated as evidence for the Cryo-EM claim.
It supplies the execution environment; the claim remains blocked because the
authors' exact checkpoint and data realization are unavailable.

## Reproduction source

The claim producers live under reproduction/:

- reproduce_mds.py — controlled Gaussian core and source checks.
- neural_npe_upgrade.py — frozen Gaussian and OUP neural posteriors.
- audit_dimension_general_proof.py — proof dependency and dimension audit.
- campaign/theorem_4_1_counterexample.py — Theorem 4.1 counterexample.
- campaign/theorem_4_2_counterexample.py — Theorem 4.2 counterexample.
- campaign/gaussian_full_comparators.py — four-method Gaussian comparator.
- campaign/cryo_full_reproduction.py — exact-scale Cryo-EM route.
- campaign/cryo_figure4_reconstruction.py — vector-source reconstruction.
- campaign/cryo_falsification_audit.py — qualification of Cryo alternatives.
- test_reproduction.py — cumulative regression checks.

Generated outputs under outputs/full and .openresearch/artifacts are retained
as evidence. They are not source material and are not silently regenerated
when a reader opens the repository.

## Formal proof audit boundaries

The proof certificate reports:

- ten robustness assumptions and four consistency assumptions were counted;
- every dependency link in R0→R1→R2→R3 and C1→C2→C3→C4 is present;
- a harmless sup k notation issue is repaired to sup absolute-value k;
- the consistency result applies to exact regular conditionals and excludes
  NPE/decoder approximation error;
- the certificate does not imply global robustness or validate imported
  theorems beyond their stated use.

These boundaries are why the theorem counterexamples and the positive
implementation checks are reported as separate verdicts.

## Evidence classes

| Class | Examples | Interpretation |
| --- | --- | --- |
| Primary source | PDF, TeX, paper figure, theorem text | What the paper actually states |
| Official code | Pinned rff.py and submodule commit | What the authors' implementation contains |
| Reproduction code | reproduction/ and campaign scripts | What this audit executes |
| Independent check | formal checkers, test_reproduction.py, verifier outputs | Whether a result can be reconstructed |
| Control | negative controls and scope checks | Whether an alternative explanation or boundary is exposed |
| Generated evidence | CSV, JSON, figures, logbook pages | Results produced by the recorded routes |

Never cite a generated value without also naming its producer, checker, and
scope boundary.
