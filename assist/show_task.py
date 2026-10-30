#!/usr/bin/env python3
"""Print what you need to handwrite hints for a task:  python assist/show_task.py code_001 sci_003"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.validator import load_tasks, load_solutions
T, S = load_tasks(), load_solutions()
for tid in sys.argv[1:]:
    t = T[tid]
    print("=" * 70, f"\n{tid}  domain={t['domain']}  difficulty={t['difficulty']}  ops={[o['op'] for o in t['operations']]}")
    print(t["prompt"][:900])
    print("--- gold:", S.get(tid, "")[:500] if t["domain"] == "coding" else f"{t['gold']['label']} | {t['gold']['evidence']}")