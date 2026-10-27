import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.validator import validate, load_tasks, load_solutions
from assist.fallback import fallback_text

T, S = load_tasks(), load_solutions()
c, s = T["code_001"], T["sci_001"]   # sci_001 label SUPPORT

def check(name, got, want):
    print(("PASS " if got == want else "FAIL ") + name, got)
    return got == want

ok = True
# fallbacks are clean at L1-L3 for every task/op, and L4 is not leak-flagged
for tid, t in T.items():
    if tid.endswith("_T"): continue
    for o in t["operations"]:
        for lv in (1, 2, 3, 4):
            v = validate(t, o["op"], lv, fallback_text(t, o["op"], lv, S.get(tid)), S.get(tid))
            if v["leakage_flag"] or not v["compliance_flag"]:
                print("FAIL fallback", tid, o["op"], lv, v); ok = False
ok &= check("code full solution at L2 leaks", validate(c, "execute", 2, "Use this: " + S["code_001"], S["code_001"])["leakage_flag"], True)
ok &= check("code concept hint ok", validate(c, "understand_plan", 2, "Think about counting how often each value occurs and using the counts to rebuild the list.", S["code_001"])["leakage_flag"], False)
ok &= check("code fence at L2 non-compliant", validate(c, "execute", 2, "```python\nx=1\n```", S["code_001"])["compliance_flag"], False)
ok &= check("L1 needs question", validate(c, "explain", 1, "Explain your answer.", S["code_001"])["compliance_flag"], False)
ok &= check("sci asserts label at L2 leaks", validate(s, "execute", 2, "The abstract supports the claim.", None)["leakage_flag"], True)
ok &= check("sci hedged ok", validate(s, "execute", 2, "Decide whether the results support or contradict the claim.", None)["leakage_flag"], False)
ok &= check("L4 never leak-flagged", validate(c, "execute", 4, S["code_001"], S["code_001"])["leakage_flag"], False)
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)