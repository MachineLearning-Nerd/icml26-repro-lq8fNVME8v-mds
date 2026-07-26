# Claim 6 source audit

Retrieval date: 2026-07-25

## Primary source and hashes

- ar5iv URL: `https://ar5iv.labs.arxiv.org/html/2602.09161`
- ar5iv HTML SHA-256:
  `f4d5f3cb369729e3d083fb09260d64e53127868dd25bae01be235f6a2bea4`
- arXiv e-print URL: `https://export.arxiv.org/e-print/2602.09161`
- e-print archive SHA-256:
  `74473bd7156ee81883e35dff79d6948657752ffae47f79ca9ed0a85a1d9ded3e`
- extracted TeX SHA-256:
  `ff81fd973e3bcba86fb23e9a0c102ec88e240f62361315c7875de54e29ea4fd2`
- primary Cryo-EM metric figure PDF SHA-256:
  `f8c4486713db2419b8c6133356dcc13ede0c8672c3d94d98aa6577820873249f`

## Anchors and exact domain

- Main statement: TeX lines 423–433, section `\label{exp:cryo-em}` and figure
  `\label{fig:main-5}`.
- Appendix task definition: TeX lines 1157–1164,
  `\label{app:cryo-em}`.
- The main text defines 32×32 images (1024 dimensions), one-dimensional shape
  inference, and replacement of an epsilon fraction by Gaussian noise.
- The appendix fixes HSP90, a discrete uniform prior over 20 conformational
  indices, 100 images per dataset, six summary coordinates (mean and standard
  deviation of per-image skewness, kurtosis, and intensity range), and 15,000
  NPE training datasets.
- The checked-in task config fixes 100 test datasets, 2,000 posterior samples,
  epsilon in `{0,.1,.2,.3,.4,.5}`, and 1,024 RFFs.
- The official benchmark fits RFF-MDS on the first 10,000 Cryo training
  datasets and reserves 5% for calibration.

## Source discrepancies

The paper's general RFF implementation paragraph says 512 features for all
experiments, but `configs/cyro_em_config.yaml` says 1,024. This reproduction
uses 1,024 because that is the task-specific checked-in Figure implementation.
The discrepancy remains evaluator-visible and is not described as exact
agreement with the generic paragraph.

The checked-in simulator config points to
`./test_time_sbi/external/cryoSBI/...`, a path absent from the released tree.
The same pinned HSP90 tensor exists at
`external/cryoSBI/tests/models/hsp90_models.pt`; the runner resolves only that
path and records both hashes/configs.

The official task discards its `seed` argument. The reproduction seeds Python,
NumPy, and Torch before every deterministic simulation stream and requires a
same-seed bit-identity/different-seed-change audit.

## Pinned code

- Official implementation commit:
  `45158124f0cbdc2f6c1ac602c9fc5501dce20af3`
- cryoSBI commit:
  `8e5832ecda626e9ab58d18cb215b5db6789533ee`
- task config SHA-256:
  `3e93b4ea02e61f21fa715c75adcf755433cd5231282e1974fe66b812ca288bfc`
- simulator config SHA-256:
  `88f6df004553d3a8daf256b6ece0e572cbf10ceb69e02c0cedcd6f3b8008873c`
- task module SHA-256:
  `a5d39645cb66bfbbfe0dc190e1fa2ff685c56e7d417729934ab3818b70440d12`
- HSP90 model tensor SHA-256:
  `218c48ca3b409124960f735304863a9b15406815ea08afe01a0bdea1fb9aeb22`
