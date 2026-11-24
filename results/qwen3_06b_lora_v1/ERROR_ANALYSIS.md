# P4 paired error analysis

Source: the actual 618-row `evaluation/paired_error_cases.csv` and paired outputs from the Colab archive, joined to the frozen prepared test prompts. The machine categories below mean **rule-validator acceptance** (`compliance AND not leakage`); they are not semantic judgments.

## Validator case counts

| Pair category | Count |
|---|---:|
| Both pass | 610 |
| Base fails, LoRA passes | 5 |
| LoRA fails, base passes | 3 |
| Both fail | 0 |

## Representative cases from actual outputs

### Both pass

`example_id=2c7ed2e38e061fa49c5d`, coding / debug-verify / level 1. Both models return: “Which part of this are you least sure about?” This matches the one-question metacognitive-cue instruction and both pass the rule validator. It is a useful format example, though identical outputs provide no evidence of a quality difference.

### Base fails, LoRA passes: solution leakage caught by rules

`example_id=1e5cfe7d9e5829b3bfec`, coding / explain / level 2. Base says to use `n*(n+1)/2`, which triggered `solution_ngram` leakage. LoRA instead names the arithmetic-series strategy and passes. The rule correctly flags the base output's answer-like formula, although the formula as phrased is also questionable for summing even numbers over an arbitrary interval; the validator cannot determine that mathematical issue.

`example_id=8f83a85ba59813d146ca`, coding / execute / level 2. Base's “Elementwise operation on tuples.” was flagged `too_short`. LoRA's suggestion to use tuple unpacking passed, but appears mismatched to the requested elementwise AND operation: unpacking alone does not perform the elementwise operation. This is a likely semantic false negative in the rule validator and needs human confirmation.

### LoRA fails, base passes: final answer disclosed at level 2

`example_id=5c785c68ba210046a618`, coding / execute / level 2. Base gives a generic approach. LoRA says the function should return `[1]`, which is the exact expected result in the provided test and violates “Never reveal the final answer.” The automated rule flags the LoRA output as `code_present`; that category catches the violation but does not describe it clearly as answer leakage / excessive assistance.

`example_id=696592722561d7c6f04a`, coding / execute / level 2. LoRA's “Swap the two numbers.” was flagged `too_short`; base's “Swap the two numbers by exchanging their values.” passed. Both are weak hints, but the validator only detects the length issue.

### Both fail

There were **no** pairs in this category. Do not invent a both-fail example from these 618 outputs. Other both-pass pairs could still contain semantic errors the rules did not detect.

## What rules miss or mislabel

- `example_id=63d40f95688bb9afe455` (both pass; coding / debug-verify / level 3): LoRA advises verifying all values are integers or floats. The task only asks whether all tuple elements share a type; they could all share another type. This is likely incorrect guidance despite validator acceptance. Base's steps are also largely tautological. This is an assistant text review, not a human rating.
- `example_id=0810b079cd0968dfb967` (both pass; understand-plan / level 2): both responses merely repeat “swap the two numbers,” with no actionable plan. The rule validator accepts both because its scope is format/compliance and leakage checks, not usefulness.
- `example_id=427934499801daae296d` (base fails, LoRA passes): base echoes the level instruction inside its response and was flagged `prompt_echo`; LoRA avoids the echo but only restates the task.
- `example_id=91584b2ff7f4f4fe9cdd` (base fails, LoRA passes): base is ordinary prose describing magnitude and angle, but was flagged `code_present`; this looks like a possible false positive. Inspect the validator rule before treating it as a real compliance error.
- `example_id=a5e99af90a42170a4307` (base fails, LoRA passes; science / execute / level 4): base labels the claim SUPPORTS even though the abstract says CX3CR1 signaling promotes Th2 survival and blocking it reduces disease; LoRA returns CONTRADICT. The base answer is factually wrong and LoRA's label is consistent with the abstract. The rule categorized the base issue as `no_answer`, which obscures the actual wrong-label failure.

These observations come from direct reading of model outputs by the assistant and should be treated as **qualitative, unblinded error inspection**, not as independent human review or the pending blinded ratings. The two-rater quality evaluation remains unfilled. A human reviewer should verify these interpretations before the paper calls them semantic error findings.
