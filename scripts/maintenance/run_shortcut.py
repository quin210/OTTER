"""Seed-0 re-validation with --lexical-shortcut (cost option): same edition, LLM skipped when claim/evidence overlap did not drop."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_simulation as RS
s = RS.stage_revalidate(0, tag="_shortcut", lexical_shortcut=True)
print(json.dumps(s["cost"], indent=1))
