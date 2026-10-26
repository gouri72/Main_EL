# P3 data pipeline (run from the repo root, Windows PowerShell)

    python data\scripts\download_data.py                      # MBPP + SciFact -> data\raw\
    python data\scripts\build_coding.py  --n 5 --seed 42      # Day-1 sample (use --n 150 for the full set)
    python data\scripts\build_science.py --n 5 --seed 42
    python data\scripts\merge_tasks.py                        # merges + validates + runs coding solutions

Files written: data\tasks_coding.jsonl, data\tasks_science.jsonl, data\solutions.jsonl, data\tasks.jsonl

Conventions on top of the frozen schema (announce to team):
- Transfer variants are lines in tasks.jsonl with id "<id>_T"; a variant's transfer_task_id is its own id.
- Reference solutions live in data\solutions.jsonl ({task_id, solution}); P4's leakage check reads it.
Known limitations (go in the paper's Data limitations): coding variants are automatic near-transfer
(renamed function + shifted integer inputs); operation labels, difficulty and error types are heuristic.
