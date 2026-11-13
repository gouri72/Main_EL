# P3 Transfer Quality Audit

This report contains automated structural checks.
It is not a human semantic-validity assessment.

## Dataset

- Coding base tasks: 150
- Science base tasks: 150
- Coding transfer pairs: 150
- Science transfer pairs: 150
- Shared science base-abstract groups: 23

## Operation coverage

| Domain | Operation | Base tasks |
|---|---|---:|
| coding | understand_plan | 150 |
| coding | execute | 150 |
| coding | debug_verify | 80 |
| coding | explain | 74 |
| science | understand_plan | 150 |
| science | execute | 150 |
| science | debug_verify | 111 |
| science | explain | 150 |

## Operation diversity by difficulty

| Domain | Difficulty | Distinct operation lists |
|---|---:|---:|
| coding | 1 | 4 |
| coding | 2 | 4 |
| coding | 3 | 4 |
| science | 1 | 2 |
| science | 2 | 2 |
| science | 3 | 1 |

## Potential issues

These are diagnostic flags, not automatic failures.

| Check | Count | Example task IDs |
|---|---:|---|
| No issues flagged | 0 | - |

## Interpretation

- Coding variants mainly assess near transfer.
- Equal science labels do not establish equal reasoning skill.
- Shared science abstracts require leakage-safe sampling.
- Operation diversity does not establish label validity.
- Human semantic auditing is still required.
