"""Prompt templates for hint generation. Bump PROMPT_VERSION whenever you edit anything here (it is part of the cache key)."""
import re

PROMPT_VERSION = "v6"

SYSTEM = ("You are a careful tutor writing assistance for a student. Follow the level instructions exactly. "
          "Output only the assistance text: no preface, no labels such as 'Hint:' or 'Level 2', and do not repeat the task.")

# Rules per (domain, level). Level 4 and the science levels 2-3 differ by domain; mixing them in one string made the model
# answer coding tasks in the science format.
RULES = {
 ("any", 1): "Level 1 (metacognitive cue): reply with EXACTLY ONE short question (one sentence ending in '?') that makes the student "
             "reflect on their own process, for example 'Which part of this are you least sure about?' or 'How could you check your answer?'. "
             "Do not answer the question. Do not mention any specific method, formula, code or answer. Max 20 words.",
 ("coding", 2): "Level 2 (conceptual hint): name the relevant concept or strategy in plain words, in one or two sentences. "
                "Do not write code. Never reveal the final answer. Max 40 words.",
 ("coding", 3): "Level 3 (partial scaffold): give a numbered plan of 3-4 steps and leave the key step for the student to do. "
                "Do not write the full solution. Max 80 words.",
 ("coding", 4): "Level 4 (full assistance): write the complete, correct Python solution in one code block, based on the "
                "reference solution. No explanation outside the code block.",
 ("science", 2): "Level 2 (conceptual hint): in one or two sentences, say WHERE the student should look: which part of the claim or "
                 "which comparison in the abstract, for example 'Look at how the two groups differ on the main outcome.' "
                 "Describe where to look, never what the abstract shows or whether the claim holds. Max 40 words.",
 ("science", 3): "Level 3 (partial scaffold): give a numbered plan of 3 steps. Every step is an ACTION for the student, for example "
                 "'1. Split the claim into its parts. 2. Find the abstract sentences about <topic of the claim>. 3. Compare what they say with the claim.' "
                 "Use the claim's own topic words. Never describe what the abstract shows and never say whether the claim holds. Max 60 words.",
 ("science", 4): "Level 4 (full assistance): reply in exactly this format: "
                 "'Correct label: <SUPPORT or CONTRADICT>. Evidence: <one or two sentences from the abstract>.'",
}

# Written from the STUDENT's point of view so the model does not answer on the student's behalf.
OP_FOCUS = {
 "understand_plan": "The student is trying to understand the task and plan an approach.",
 "execute": "The student is carrying out their plan (writing the solution / locating the evidence).",
 "debug_verify": "The student is checking their answer and looking for mistakes.",
 "explain": "The student is explaining their own reasoning for their answer.",
}

SCIENCE_NO_LEAK = ("Do not say whether the abstract supports or contradicts the claim, and do not say the claim is true or false. "
                   "Do not mention sentence numbers or brackets such as [3].")

# Retry feedback: tells the model why its previous reply was rejected (keys = validator reasons, LEAK:<cause>).
FIXES = {
 "LEAK:label": "Do not say that the abstract supports or contradicts the claim, or that the claim is true or false.",
 "LEAK:sentence_ref": "Do not mention sentence numbers or brackets like [2].",
 "LEAK:evidence_ngram": "Do not copy wording from the abstract; stay general and use your own words.",
 "LEAK:solution_ngram": "Do not write any part of the solution code.",
 "not_a_question": "Reply with exactly one question and nothing else.",
 "too_long": "Make it shorter.",
 "too_short": "Write at least one full, specific sentence.",
 "no_answer": "Include the complete answer in the required format.",
 "code_present": "Do not include code.",
 "unclosed_code_block": "Write complete, valid Python inside a closed code block.",
 "code_syntax_error": "Write complete, valid Python inside a closed code block.",
 "prompt_echo": "Do not repeat the instructions.",
}

def feedback_text(reasons):
    msgs = []
    for r in reasons:
        m = FIXES.get(r)
        if m and m not in msgs: msgs.append(m)
    return ("Your previous reply was rejected. " + " ".join(msgs)) if msgs else ""

def build_user_prompt(task, op, level, solution=None, feedback=""):
    dom = task["domain"]
    body = task["prompt"]
    if dom == "science":
        body = re.sub(r"\[\d+\]\s*", "", body)[:2500]     # remove sentence markers so the model has no numbers to cite
    rule = RULES.get(("any", level)) or RULES[(dom, level)]
    extra = ""
    if level == 4:   # full assistance may see the gold answer
        if dom == "coding" and solution:
            extra = "\nReference solution:\n" + solution
        elif dom == "science":
            extra = f"\nCorrect label: {task['gold']['label']}\nEvidence: " + " ".join(task["gold"]["evidence"])[:600]
    guard = f"\n{SCIENCE_NO_LEAK}" if (dom == "science" and level <= 3) else ""
    fb = f"\n{feedback}" if feedback else ""
    return f"Task:\n{body}\n\n{OP_FOCUS[op]}\n{rule}{guard}{extra}{fb}"