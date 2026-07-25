# Claim 5 limitations and deviations

- The exact paper prose is qualitative (“in general”). The stronger imported
  judge gloss is the only proposition eligible for falsification here.
- The run reproduces RFF-MDS, NPE, NNPE, and NPE-OR, directly answering the
  judge's missing-comparator criticism. It does not retrain decoder-MDS,
  NPE-RS, or NPE-PFN, none of which is needed to decide the imported
  NPE/NNPE/NPE-OR comparator proposition.
- RFF transformation is streamed to control memory. Feature construction,
  seed, dimension, and mean embedding are unchanged.
- The authors' Gaussian config says `n_test: 500`, while the paper and
  evaluation text specify 100 test datasets. This reproduction follows the
  paper's exact evaluation quantifier of 100.
- The official configuration-generation default uses outlier shift 2, while
  the paper explicitly specifies shift 3 for the proportion sweep. This
  reproduction follows the paper.
- One fixed deterministic seed is used for the full 100-dataset design.
  Uncertainty is over paired test datasets, as specified for the paper's error
  bars, rather than over independently retrained neural estimators.
