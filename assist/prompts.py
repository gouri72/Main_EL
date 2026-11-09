"""Prompt templates for hint generation. Bump PROMPT_VERSION whenever you edit anything here (it is part of the cache key)."""
PROMPT_VERSION = "v5"

SYSTEM = ("You are a careful tutor writing assistance for a student. Follow the level instructions exactly. "
          "Output only the assistance text: no preface, no labels such as 'Hint:' or 'Level 2', and do not repeat the task.")

LEVEL_RULES = {
 1: "Level 1 (metacognitive cue): reply with EXACTLY ONE short question (one sentence ending in '?') that makes the student "
    "reflect on their own process, for example 'Which part of this are you least sure about?' or 'How could you check your answer?'. "
    "Do not answer the question. Do not mention any specific method, formula, code or answer. Max 20 words.",
 2: "Level 2 (conceptual hint): name the relevant concept or strategy in plain words, in one or two sentences. "
    "Do not write code. Never reveal the final answer or the conclusion. Max 40 words.",
 3: "Level 3 (partial scaffold): give a numbered plan of 3-4 steps and leave the key step for the student to do. "
    "Do not write the full solution and never state the final answer or conclusion. Max 80 words.",
 4: "Level 4 (full assistance): give the complete answer. For a coding task, write the full correct Python solution "
    "in a code block, based on the reference solution. For a science claim, reply in exactly this format: "
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

def build_user_prompt(task, op, level, solution=None):
    body = task["prompt"]
    if task["domain"] == "science":
        body = body[:2500]   # keep the 0.6B context small
    extra = ""
    if level == 4:   # full assistance may see the gold answer
        if task["domain"] == "coding" and solution:
            extra = "\nReference solution:\n" + solution
        elif task["domain"] == "science":
            extra = f"\nCorrect label: {task['gold']['label']}\nEvidence: " + " ".join(task["gold"]["evidence"])[:600]
    guard = f"\n{SCIENCE_NO_LEAK}" if (task["domain"] == "science" and level <= 3) else ""
    return f"Task:\n{body}\n\n{OP_FOCUS[op]}\n{LEVEL_RULES[level]}{guard}{extra}"