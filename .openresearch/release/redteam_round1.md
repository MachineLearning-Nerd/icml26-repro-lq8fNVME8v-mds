# Evaluator-blind pre-publication review — round 1

Status: **PASS**

The reviewer was given only the fresh candidate directory and started
from `README.md`, `logbook.json`, and `pages/index.md`. No internal
OpenResearch paths, run dashboard, or unpublished branch knowledge was
used to locate evidence.

## Claim conclusions

| Claim | Located verdict | Canonical file |
| --- | --- | --- |
| Claim 1 | VERIFIED | `pages/current-method-regression/page.md` |
| Claim 2 | VERIFIED | `pages/current-method-regression/page.md` |
| Claim 3 | FALSIFIED | `pages/current-theorem-4-1/page.md` |
| Claim 4 | FALSIFIED | `pages/current-theorem-4-2/page.md` |
| Claim 5 | FALSIFIED | `pages/current-gaussian-comparators/page.md` |
| Claim 6 | BLOCKED | `pages/current-cryo-em/page.md` |

## Files opened

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `README.md` | 1043 | `e988be2ec0de390c5c4b25671f9ae3fa00d360685f182c9ed5a612cb078e4d5f` |
| `logbook.json` | 3434 | `3bcaac5372b54ce0de54a37aeb2f4f22b7d7db9bae69777b51931532347ef549` |
| `pages/00-scored-evidence-summary/page.md` | 267630 | `d133582364932757cdf93c18a2eead9bda2b6a500a9f29a6ea1d016b8d3cb95b` |
| `pages/claim-1-plug-in-mmd-summary-independent-of-npe/page.md` | 2072 | `24764760c5b06c6b83093aec7d0ae85c5a24efe355f74292dfe6cd4c36e6bf76` |
| `pages/claim-2-rff-efficiency-and-lightweight-adaptation/page.md` | 366596 | `16674b059b94f0b1ea680dbfb05939ed0438bf55cace4f303077930e7d9f4641` |
| `pages/claim-3-robustness-gains-with-low-overhead/page.md` | 2870 | `a39f1f90d98480fbe1036ebb1d8d2b4f673c3b806ad6ecb6468eebb7cbbfad57` |
| `pages/claim-4-robustness-and-consistency-guarantees/page.md` | 530960 | `56d5c7a3c23b6149b75412343ecd3bcd85bd9af4251408b12effb09be0ccf044` |
| `pages/current-cryo-em/attempts.md` | 3880 | `1a59d123b67a553a95b01cafd3d3c7cab400e493bc8cbdba4b6917b2d5b419e7` |
| `pages/current-cryo-em/check_cryo_falsification.py` | 4069 | `a3ffe52df7513d0ab0fd1a5e059978701d7de63d17493eb9d7a5591df1c0ba55` |
| `pages/current-cryo-em/cryo_falsification_audit.py` | 7178 | `488437ee1583c738fb5a9ba1cb8e950bfd757ee48aedfbf51b55a047cac42dc5` |
| `pages/current-cryo-em/cryo_full_reproduction.py` | 30822 | `47a88031771ac18e54a6e15e682a156767bb40fdfb91421859a55b05a9bf13fc` |
| `pages/current-cryo-em/falsification_audit.json` | 4090 | `f8e5539b03a284f6b1ca595ec7043574c01e002bd83c4d1f7d83e0768147d6c5` |
| `pages/current-cryo-em/figure4_digitized.csv` | 5864 | `453d8ff35943196b44ba8b2fc084b809b15ef32ba4e220f2779e664162125103` |
| `pages/current-cryo-em/independent_checker_output.json` | 1054 | `290d3f3d3d5fff6057cf8b7ae1d4eb4a68246370d8fca48e12bc379f583e7214` |
| `pages/current-cryo-em/negative_control_output.json` | 604 | `f706f716d54061724764f15ec3a9a628fd7823bb0e60e54eb2907f4d6e60521b` |
| `pages/current-cryo-em/page.md` | 4746 | `bc6bd432bc843b701d2d0f94664ad518f7536c6bb17f5ef8aa9cf90a8978b8ea` |
| `pages/current-cryo-em/route_1_raw_trials.csv` | 127915 | `ef27c8f5d79c1cc225d1fba328c3f4de86f0b3bf914b7da1de47797dac3150de` |
| `pages/current-cryo-em/route_1_summary.json` | 12156 | `f80cff6ce127a5c8000aa712461f48a7e478202d97273b276e75eac71068d70c` |
| `pages/current-cryo-em/runtime.json` | 380 | `f4de6315be230983fc730b1fc23a9881d39ad7107fc9ff3d52fc2de067aa709b` |
| `pages/current-cryo-em/verifier_output.json` | 410 | `d12de86bd2f0beb936b08e466bc36ceee964969daa009dea39ab12c72398e5ef` |
| `pages/current-cryo-em/verify_cryo_falsification.py` | 2126 | `622381cf1b557ea036ff369660ceced69f478bf8f012df729ae8fbe241e72053` |
| `pages/current-gaussian-comparators/aggregate.csv` | 3375 | `2f94f70e25ca6b114d34af14cfb00c7c303afdf7bb871a851602a7adf6d8e91f` |
| `pages/current-gaussian-comparators/check_gaussian_comparators.py` | 4540 | `c70ec0e9341de18a7cca89daadfeabe03c166d91f14556248cdd06c0eab3f49e` |
| `pages/current-gaussian-comparators/gaussian_full_comparators.py` | 20074 | `4a78274728a67ecb07a8423779efb302db1125fb284f1a1bbfd6c1362dde48c3` |
| `pages/current-gaussian-comparators/independent_checker_output.json` | 1289 | `b5b5f4be85922c2622f58414444140a675f887a35cf10cfed11b077151a65839` |
| `pages/current-gaussian-comparators/negative_control_output.json` | 563 | `39bc6bd4178d0aa798ab2733a86cd3f999898d166db16847af4d9d0c9efdfd31` |
| `pages/current-gaussian-comparators/page.md` | 4351 | `abdf949cb8e82b0e4ee603b53655cd9f8b5f55fcda8232da458b188c54532775` |
| `pages/current-gaussian-comparators/raw_trials.csv` | 245833 | `3b813e9dc810b8b82abaac4631da3765e87951bcfbd1ad557d491697788c91bb` |
| `pages/current-gaussian-comparators/runtime.json` | 278 | `dd16986445c42536f012f0b51e98eeeebe4e89b32b32c797eb02e4e9c5f63e5d` |
| `pages/current-gaussian-comparators/summary.json` | 12416 | `0550a906de402b4ab8846cdf3ade0eca2f7ef46a8e3aec27450faf88d5b4e50e` |
| `pages/current-gaussian-comparators/verifier_output.json` | 577 | `0e1431b5e248356de074ecc59b7951726f204a3f02b01b933f4246f27d377354` |
| `pages/current-gaussian-comparators/verify_gaussian_claim.py` | 3853 | `5ac15e60ef4cef0e9bca76048393f42d887c0ba331172bf5b1aaebeed9e6fa33` |
| `pages/current-method-regression/accepted_metrics.json` | 1474 | `3f0bb04fcaad20cfc7a3cc19a820cb4b35125f19cfc68f3aae463794dd2acc88` |
| `pages/current-method-regression/claim1_checker_output.json` | 332 | `f8a1e4636acb98d75961726e12a5004c7008df9bd72f55cbe24ba6c8084ff32f` |
| `pages/current-method-regression/claim1_negative_control_output.json` | 254 | `160b4ebb6927f3beb4203c100a34ea75f6023d595474c7e494ae6494171d19b7` |
| `pages/current-method-regression/claim2_checker_output.json` | 394 | `1568e81285cca3ee95a6b6e790a32e3ee66a8a68ec428b59984d5db30fe2ba2c` |
| `pages/current-method-regression/claim2_negative_control_output.json` | 294 | `2e9db0369819ca1ebd5e8b96008750090c39d7ff66bbdcac4c2b75ccc83677bb` |
| `pages/current-method-regression/cumulative_runtime.json` | 1325 | `43ff65dca7a5e965b73d85161a0531b6cfb50de65f7fdf214d314f0b8edd9af6` |
| `pages/current-method-regression/integrate_neural_upgrade.py` | 3988 | `b62fccac3e50c3cc89a4811f7e9a3c94ea09ecfdcf2a72e7c2ee6a835c4b3e1d` |
| `pages/current-method-regression/neural_npe_upgrade.py` | 22693 | `0ab583506b80670aa2bc75b3736ced9be2a29f35aecc93f7d5910cc203bf620b` |
| `pages/current-method-regression/page.md` | 3559 | `6ec673a04630e5fa96413a435424b503c0ed712209a7a01da0bc97f203337441` |
| `pages/current-method-regression/reproduce_mds.py` | 26375 | `4f19f015a19bb588da046ee2d68c77b974cbeb11325cc511eaadcddf750927e5` |
| `pages/current-method-regression/test_reproduction.py` | 9562 | `26766567b2598ed7e22ddba440fce38cae83fa1b166c42bee9a14933b2e4324b` |
| `pages/current-method-regression/verify_claim1.py` | 509 | `b81deef908988292fb677b355f196e964a3fd9b848d088b7b2ff24b167a3983b` |
| `pages/current-method-regression/verify_claim2.py` | 571 | `89586d35fa313796f6baeba1889b16a68117eaeb492582d1e7b055a6c24420f3` |
| `pages/current-release-assessment/page.md` | 4872 | `90eb194c7b092dfdc9accbb92dbe17041ea8450ba30a8d194d0d896a88a51e28` |
| `pages/current-theorem-4-1/formal_results.json` | 1707 | `88a70e75e7dc26797467129db69360de3ad7255ef48e7c0331d318f741f8ca14` |
| `pages/current-theorem-4-1/independent_checker_output.json` | 583 | `6e4cd10878c0ed6d8741d687eaafeee0a86b9420640487a3fc2a01d81a134c1e` |
| `pages/current-theorem-4-1/negative_control_output.json` | 441 | `c74d3ad0f9b905e58eadeb56e63d3ff966fc64ed6e7d38039e31e8d1e67022a0` |
| `pages/current-theorem-4-1/page.md` | 4416 | `94e1086d9751e4bd3630a08e522e8963fb6d99e360f25d060bafccdfe831c94d` |
| `pages/current-theorem-4-1/runtime.json` | 658 | `c288a4adf94507fa8fcb2f543309575b9a66a21ab260d9c7f70b16d276471a17` |
| `pages/current-theorem-4-1/verify.py` | 2229 | `40b65a67030548df0031a12236258f22895d01ea2553e6b43187c66a8e4c68c5` |
| `pages/current-theorem-4-2/formal_result.json` | 805 | `8e9278138f265867d3d617f03d6f9c029aba3ce209557acc6188928e49f388c5` |
| `pages/current-theorem-4-2/formal_results.csv` | 3239 | `a5e401fba2f770761a26b667970dea74b958cd8db89062512ef5933705ad7742` |
| `pages/current-theorem-4-2/independent_checker_output.json` | 518 | `afe316dbe27d0bb17597e153c96bea48da95178b6670a53cb1961d609c1335e3` |
| `pages/current-theorem-4-2/negative_control_output.json` | 378 | `58df097fa65ac3a4c6ecac6bfc10d9405dab49ddb35507fd59075d718d838d44` |
| `pages/current-theorem-4-2/page.md` | 4751 | `82f96f4c908aacf0cf363415238ce7182d15094f47847e978c138895b647a9b2` |
| `pages/current-theorem-4-2/runtime.json` | 657 | `4a7167f61e66b4f862772d73b1f367d9c45a102ae36aa5afc730a485928242df` |
| `pages/current-theorem-4-2/verify.py` | 2684 | `7b7b25a2f538f455e9691442e03fc9d8174c7f5f212459b67cef273cb8ee1f92` |
| `pages/index.md` | 1696 | `923d7e7d3d13879c26e9b0ba8c6fc2436dc8a4c7a152e01ece8d8fb62fa2bb90` |
| `pages/limitations-and-falsification-attempts/page.md` | 2014 | `3f41460f199a65dff82f8062c509a0ff8adf10e54eed200217246877130659c4` |
| `pages/reproduction-protocol-artifacts-and-hashes/page.md` | 96893 | `13f3a02af265f7e0f18ca44241388731ed0eef84ba2d8f742790b0be300d59ce` |

## Conclusions that could not be verified

- None within the candidate's scientific evidence contract. The live judge score and a final published HF revision are intentionally not verifiable before publication.

## Reviewer assessment

The current verifier is obvious because current pages appear before
the section labelled exactly “Historical rejected baseline.” Claims
1–5 expose exact claims, inline numbers, downloadable machine-readable
evidence, executable code, independent checking, controls, limitations,
fixed command and compute receipts. Claim 6 exposes the same categories
but correctly remains BLOCKED after four routes.
