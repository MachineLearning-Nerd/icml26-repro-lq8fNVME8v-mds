#!/usr/bin/env python3
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "raw_results.json"
data = json.loads(path.read_text())
claim = data["claim_2"]
assert claim["verdict"] == "VERIFIED"
assert claim["paper_rff_dimension"] == 512
assert claim["gaussian"]["final_embedding_mse"] < 1e-4
assert claim["gaussian"]["median_adaptation_ms"] > 0
assert claim["oup"]["nonzero_contamination_median_adaptation_ms"] > 0
assert claim["negative_boundary"]["rff_exact_summary_gap"] > 1
print("VERIFIED")
