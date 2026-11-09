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
ok &= check("sci sentence number leaks", validate(s, "execute", 2, "Look closely at sentence 3 of the abstract.", None)["leakage_flag"], True)
ok &= check("sci ordinal sentence leaks", validate(s, "execute", 3, "The 2nd sentence matters most here.", None)["leakage_flag"], True)
ok &= check("sci step numbers ok", validate(s, "understand_plan", 3, "Plan: (1) list the claim parts, (2) find matching abstract sentences, (3) compare them.", None)["leakage_flag"], False)
ok &= check("L4 never leak-flagged", validate(c, "execute", 4, S["code_001"], S["code_001"])["leakage_flag"], False)
ok &= check("L4 prompt echo rejected", validate(c, "execute", 4, "Write the hint now.", S["code_001"])["compliance_flag"], False)
ok &= check("L4 coding without code rejected", validate(c, "execute", 4, "Hint: Make sure to initialize the buckets correctly and iterate through the list to count.", S["code_001"])["compliance_flag"], False)
ok &= check("L4 coding with code ok", validate(c, "execute", 4, "```python\n" + S["code_001"] + "\n```", S["code_001"])["compliance_flag"], True)
ok &= check("L4 science without label rejected", validate(s, "execute", 4, "Think carefully about the abstract and its results overall.", None)["compliance_flag"], False)
ok &= check("L1 too short rejected", validate(c, "explain", 1, "Why?", S["code_001"])["compliance_flag"], False)
ok &= check("L1 stray ' ?' rejected", validate(c, "explain", 1, "Why is the counting sort correct? It works by counting occurrences of each number and then placing them back. ?", S["code_001"])["compliance_flag"], False)
ok &= check("L1 two questions ok", validate(c, "explain", 1, "How could you check your answer? Which case might break it?", S["code_001"])["compliance_flag"], True)
ok &= check("unclosed code fence rejected", validate(c, "execute", 4, "```python\ndef counting_sort(my_list):\n    return sorted(my_list)", S["code_001"])["compliance_flag"], False)
ok &= check("code syntax error rejected", validate(c, "execute", 4, "```python\ndef counting_sort(my_list:\n    return 1\n```", S["code_001"])["compliance_flag"], False)
from assist.generate_hints import clean
ok &= check("clean collapses '? ?'", clean("What is the correct approach to ensure it works? ?"), "What is the correct approach to ensure it works?")
ok &= check("clean drops junk after fence", clean("```python\ndef f(x):\n    return x\n``` for i in range(len(my_list)):\n    buckets[i] += 1"), "```python\ndef f(x):\n    return x\n```")
ok &= check("L4 science label, evidence containing 'or', ok", validate(s, "execute", 4, "Correct label: SUPPORT | Evidence: results were similar in group A or group B.", None)["compliance_flag"], True)
ok &= check("L4 science wrong label rejected", validate(s, "execute", 4, "Correct label: CONTRADICT | Evidence: the results clearly differed between groups.", None)["compliance_flag"], False)
ok &= check("neutral 'confirm the relationship' not a leak", validate(s, "understand_plan", 3, "1. Analyze the claim and abstract to confirm the relationship between the two variables. 2. Find the matching evidence.", None)["leakage_flag"], False)
ok &= check("neutral 'correct approach' not a leak", validate(s, "understand_plan", 2, "Use the correct approach to evaluate each part of the claim against the abstract.", None)["leakage_flag"], False)
ok &= check("'the answer is correct' leaks (SUPPORT label)", validate(s, "explain", 2, "The answer is correct because the study shows a clear effect in the treated group.", None)["leakage_flag"], True)
ok &= check("'the claim is true' leaks (SUPPORT label)", validate(s, "explain", 2, "The claim is true given the reported results of the trial.", None)["leakage_flag"], True)
ok &= check("leak_why reports the cause", validate(s, "execute", 2, "The abstract supports the claim, see sentence 3.", None)["leak_why"], ["label", "sentence_ref"])
ok &= check("'to support the claim' (instruction) not a leak", validate(s, "execute", 3, "1. Find the key results. 2. Focus on the findings and limitations to support the claim.", None)["leakage_flag"], False)
ok &= check("'which supports the claim' leaks", validate(s, "execute", 2, "The relevant concept is the biomarker association, which supports the claim.", None)["leakage_flag"], True)
from assist.prompts import build_user_prompt, feedback_text
pc, ps = build_user_prompt(c, "execute", 4, S["code_001"]), build_user_prompt(s, "execute", 4)
ok &= check("coding L4 prompt has no science format", "Correct label" in pc or "SUPPORT" in pc, False)
ok &= check("science L4 prompt has the label format", "Correct label:" in ps, True)
ok &= check("science prompt strips [n] sentence markers", "[3]" in build_user_prompt({**s, "prompt": "Claim: x [1] a. [3] b."}, "execute", 2).split("Do not mention sentence numbers")[0], False)
ok &= check("retry feedback names the cause", "sentence numbers" in feedback_text(["LEAK:sentence_ref"]), True)
ok &= check("no feedback when nothing to fix", feedback_text([]), "")
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)