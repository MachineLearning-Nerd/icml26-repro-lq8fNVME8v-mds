#!/usr/bin/env python3
from pathlib import Path
import json
import trackio

ROOT = Path(__file__).resolve().parents[1]
run = trackio.init(
    project="minimum-distance-summaries-repro",
    name="cpu-six-claim-reproduction",
    config={"openreview_id":"lq8fNVME8v","claims":6,"device":"cpu","gpu_used":False},
    embed=False,
    auto_log_gpu=False,
    auto_log_cpu=False,
)
artifact = trackio.Artifact(
    "minimum-distance-summaries-cpu-reproduction",
    type="dataset",
    description="Complete six-claim CPU reproduction and audit with frozen NPE checkpoints, raw paired trials, formal theorem checks, Cryo-EM qualification, source pins, tests, and boundary evidence.",
)
artifact.add_dir(ROOT / "reproduction", name="reproduction")
artifact.add_dir(ROOT / "outputs", name="outputs")
artifact.add_dir(ROOT / "source" / "arxiv", name="source/arxiv")
artifact.add_file(ROOT / "paper" / "2602.09161.pdf", name="paper/2602.09161.pdf")
logged = trackio.log_artifact(artifact, aliases=["challenge","cpu","complete"])
trackio.finish()
print(json.dumps({"artifact":logged.qualified_name,"files":len(logged.manifest or []),"size":logged.size}, sort_keys=True))
