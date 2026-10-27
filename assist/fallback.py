"""Safe-by-construction fallback hints per (domain, op, level). They never mention task content, so they cannot
leak. Level 4 is built from gold data (full assistance by definition)."""
import json

GENERIC = {
 "understand_plan": {
  1: "What is the problem really asking, and what would a correct answer look like?",
  2: "Restate the goal in your own words and list the inputs and the output you need before choosing a method.",
  3: "Break the problem into two or three smaller steps. Write the first step and leave the rest for you to fill in.",
 },
 "execute": {
  1: "Which of your planned steps are you working on right now, and what comes next?",
  2: "Carry out your plan one step at a time and check the result of each step before moving on.",
  3: "Here is a skeleton: set up the inputs, do the main loop or comparison, then return the result. Fill in the main part yourself.",
 },
 "debug_verify": {
  1: "How could you check whether your answer is right? Which case might break it?",
  2: "Test your solution on a tiny example by hand and on an edge case such as an empty or extreme input.",
  3: "Pick one failing case, trace it line by line, and note the first place where the value differs from what you expected.",
 },
 "explain": {
  1: "Could you explain to a classmate why your answer works?",
  2: "Describe in two sentences the key idea of your answer and why it handles the edge cases.",
  3: "Complete this sentence: 'This works because ... and it would fail if ...'.",
 },
}

def fallback_text(task, op, level, solution=None):
    if level == 4:
        if task["domain"] == "coding":
            return "Full solution:\n" + (solution or "(reference solution not available)")
        g = task["gold"]
        return f"The correct label is {g['label']}. Key evidence: " + " ".join(g["evidence"])[:600]
    return GENERIC[op][level]