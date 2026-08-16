# Pinned source manifest

These hashes identify the principal paper and implementation sources used by
the audit. Compute them with shasum -a 256 from the repository root.

| Source | Location or gitlink | SHA-256 or commit |
| --- | --- | --- |
| Paper PDF | paper/2602.09161.pdf | 1fc774ab166496d0861720b14212204c46cf920dc22c5df006d1d48f5eba01f0 |
| Paper TeX | source/arxiv/arxiv_main.tex | ff81fd973e3bcba86fb23e9a0c102ec88e240f62361315c7875de54e29ea4fd2 |
| Official repository | source/official-repo | 45158124f0cbdc2f6c1ac602c9fc5501dce20af3 |
| Official RFF adapter | source/official-repo/src/tt_sbi/tta/rff.py | 0a7e8a2261c9ab9f8b327c01335dd7ff9cb7d5efd48abcf1980bf9c125d329b9 |
| Cryo-EM dependency | external/cryoSBI | 8e5832ecda626e9ab58d18cb215b5db6789533ee |

## Packaging and configuration notes

- The paper-scale RFF dimension is 512.
- The official repository default is 256.
- The reproduction explicitly overrides the dimension to 512.
- The official package initializer has an undeclared sbibm import path;
  reproduction code loads the pinned self-contained RFF module directly.
- The formal run command uses the committed uv.lock and Python 3.12.
- No GPU was used by the recorded formal runs.

## Evidence source hashes

The principal generated artifacts are also listed in
[EVIDENCE_MANIFEST.json](EVIDENCE_MANIFEST.json). Those hashes are intended to
make an evidence change visible during review; a changed artifact must be
paired with a new producer/checker receipt.
