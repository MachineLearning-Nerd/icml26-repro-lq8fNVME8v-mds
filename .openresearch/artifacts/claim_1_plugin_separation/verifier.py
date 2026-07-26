#!/usr/bin/env python3
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "raw_results.json"
data = json.loads(path.read_text())
claim = data["claim_1"]
assert claim["verdict"] == "VERIFIED"
assert claim["gaussian"]["paired_queries"] >= 300
assert claim["oup"]["paired_queries"] >= 250
assert len(claim["gaussian"]["tensor_sha256_before_after"]) == 64
assert len(claim["oup"]["tensor_sha256_before_after"]) == 64
print("VERIFIED")
