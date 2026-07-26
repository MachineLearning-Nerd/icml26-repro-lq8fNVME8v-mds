# Release-critical command ledger

The fixed reproduction command on every formal experiment node was:

```text
git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py
```

All 14 runs listed by `orx runs ced26099-8388-4933-96cf-3094ef3820b9`
through scientific winner `fd04a075-47dd-487c-a947-c6972227a67b` used that
same inherited command. Each launch used the following orchestration shape,
with the experiment ID substituted:

```text
orx exp run <experiment-id> --backend hf --flavor cpu-upgrade --image ghcr.io/astral-sh/uv:python3.12-bookworm-slim --timeout 4h
orx exp wait <experiment-id> --timeout 480
orx logs <run-id>
```

The formal run IDs, newest last, were:

```text
920236b7-ee7d-495a-86a6-b445d519ccf9
c699a3c5-9a14-41d6-97bf-d53d310bc9d4
091e80cc-14ce-4949-9031-0a54d065ddf0
e50ebf39-1199-4b76-9070-a7fa83e1b342
93bf6002-6dae-414e-a678-b2b39d42f138
8df1e229-5cc6-4ecb-b6cd-0fd313fa7a9b
021dcbe0-ddcc-47cc-9bde-f131d61fa75f
cb709f65-56ee-4945-82e7-f9b7a4ff5d9f
e77d5bab-4abd-4e63-a589-10e137cbd7ae
1df5f0be-40f8-495f-bf0c-ea9ada0ebc9f
aebd0b74-2974-40d4-95f1-18db9d6534c6
05d23079-27e1-4ce2-acf4-df50c9a1d351
6e01e046-c72a-4ddc-8a83-daf8d95be54d
fd04a075-47dd-487c-a947-c6972227a67b
```

Short one-core local preparation and release checks were:

```text
uv run --frozen python reproduction/campaign/extract_logged_evidence.py fd04a075-47dd-487c-a947-c6972227a67b
uv run --frozen python reproduction/build_campaign_figures.py
uv run --frozen marimo check notebooks/mds_reproduction.py
uv run --frozen python reproduction/validate_release.py
uv run --frozen python reproduction/blind_review.py <fresh-candidate> .openresearch/release/redteam_round1.md --round 1
uv run --frozen python -m py_compile reproduction/publish_existing_space.py reproduction/validate_release.py reproduction/blind_review.py reproduction/campaign/extract_logged_evidence.py reproduction/build_campaign_figures.py
```

The authorized publication command, run only after all gates pass, is:

```text
uv run --frozen python reproduction/publish_existing_space.py --candidate .trackio/logbook --allowlist .openresearch/release/upload_allowlist.txt --manifest .openresearch/release/candidate_space.sha256 --parent a9e77b682e084d5174725eef5adfb172cb184e67 --repo-id DineshAI/lq8fNVME8v
```

Post-publication verification uses:

```text
hf download DineshAI/lq8fNVME8v --repo-type space --revision <published-revision> --local-dir <fresh-directory>
uv run --frozen python reproduction/blind_review.py <fresh-directory> <post-publication-review> --round post-publication
git ls-remote origin refs/heads/master
```
