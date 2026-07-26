# Claim 6 limitations and deviations

- The primary RMSE reproduces the paper metric exactly at full task scale.
- The secondary predictive result uses a 1,024-RFF mean-embedding distance,
  not the paper's exact five-bandwidth quadratic MMD. Computing a 15,000-image
  pairwise kernel for every method/test pair is not CPU-feasible. The result
  is labeled as approximate everywhere and cannot independently carry the
  claim verdict.
- The paper says 512 RFFs generically; the released Cryo task config says
  1,024. This run follows the task-specific config.
- Raw images are streamed/disk-backed. No dataset, state, image, feature, or
  test case is downsampled.
- The stale released HSP90 path is resolved to the exact tensor in the pinned
  cryoSBI submodule; its SHA-256 is recorded.
- Global RNGs are explicitly seeded because the released task ignores its seed
  parameter. Same-seed and different-seed controls are mandatory.
- Only NPE and RFF-MDS are necessary for the exact imported claim. NNPE and
  NPE-OR curves shown in the paper figure are not retrained here and are not
  used to establish the NPE-versus-MDS proposition.
- A single deterministic model-training seed is used. Statistical uncertainty
  is over the paper's 100 paired test datasets, matching the source error-bar
  definition, not over independently retrained NPEs.
- Route 1 follows the authors' continuous spline-flow posterior sampling even
  though the simulator parameter is inherently one of 20 discrete HSP90
  states.
- Route 2 normalizes the learned density over all 20 admissible states. This
  is exhaustive over the stated task domain but differs from the authors'
  Figure 4 sampling implementation. Both result sets remain visible.
- One stochastic full-scale retraining cannot by itself falsify the empirical
  Figure 4 claim. A missed verification threshold remains BLOCKED pending the
  required independent routes.
- Route 3 reconstructs the exact vector figure and error intervals. It is
  useful for quantifying the published target and discrepancies, but is not
  independent experimental evidence and cannot carry a VERIFIED verdict.
