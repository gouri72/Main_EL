"""Prompt templates for hint generation. Bump PROMPT_VERSION whenever you edit anything here (it is part of the cache key)."""
PROMPT_VERSION = "v4"

SYSTEM = ("You are a careful tutor writing assistance for a student. Follow the level instructions exactly. "
          "Output only the assistance text: no preface, no labels such as 'Hint:' or 'Level 2', and do not repeat the task.")

LEVEL_RULES = {
 1: "Level 1 (metacognitive cue): ask ONE question that prompts the student to reflect on their own process. "
    "Do not mention any specific method, formula, code or answer. Never reveal the final answer. Max 25 words. It must end with a question mark.",
 2: "Level 2 (conceptual hint): name the relevant concept or strategy in plain words. "
    "Do not write code. Never reveal the final answer or the conclusion. Max 40 words.",
 3: "Level 3 (partial scaffold): give a numbered plan of 3-4 steps and leave the key step for the student to do. "
    "Do not write the full solution and never state the final answer or conclusion. Max 80 words.",
 4: "Level 4 (full assistance): give the complete answer. For a coding task, write the full correct Python solution "
    "in a code block, based on the reference solution. For a science claim, state the correct label and the key evidence in 2-3 sentences.",
}

OP_FOCUS = {
 "understand_plan": "Focus on understanding the task and planning an approach.",
 "execute": "Focus on carrying out the plan (writing the solution / locating the evidence).",
 "debug_verify": "Focus on checking the answer and finding mistakes.",
 "explain": "Focus on explaining why the answer is correct.",
}

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
    return f"Task:\n{body}\n\n{OP_FOCUS[op]}\n{LEVEL_RULES[level]}{extra}"