# Datasets

We use two task domains with deliberately different label semantics: Python programming and scientific claim verification. All tasks come from public benchmarks; nothing is synthesized from scratch.

**Coding.** Tasks are drawn from MBPP (Austin et al., 2021), a benchmark of entry-level Python problems, each with a natural-language description, a reference solution and three assert-style tests. We scan the 974 problems in a seeded random order (seed 42) and keep a problem if it has no setup code, its description contains no digits, its reference solution passes its own tests and uses no file, network or process calls, its tests have the form `assert f(...) == value`, and an automatic transfer variant can be built (below). The first 150 problems that pass form the coding session tasks. The prompt is the MBPP description plus its first test, following the standard MBPP prompt format.

**Science.** Tasks are drawn from SciFact (Wadden et al., 2020), expert-written scientific claims paired with research abstracts and annotated rationale sentences. We use the labelled training and development claims, keep those whose cited evidence has a single label (693 usable claims: 456 SUPPORT, 237 CONTRADICT), and sample 75 SUPPORT and 75 CONTRADICT claims (seed 42) as session tasks. The prompt contains the claim and one cited abstract; the gold answer is the SciFact label and the annotated evidence sentences from that abstract.

**Transfer variants.** Every session task has one transfer variant used in the unaided transfer block. For coding, a variant keeps the problem but renames the function and shifts the integer literals in the test inputs; expected outputs are recomputed by running the MBPP reference solution, so variants pass the same logic by construction. For science, a variant is a different SciFact claim with the same label and the same number of evidence sentences (1, 2 or 3+), chosen as the most lexically similar unused claim from a different abstract.

**Operations, difficulty and error types.** Each task is annotated with the reasoning operations it involves (understand/plan, execute, debug/verify, explain), a difficulty bin (1 to 3) and coarse error types. Coding and science use separate label semantics and we do not treat them as equivalent. Every operation label records its source: in science, `execute` (locating evidence) is grounded in SciFact's annotated rationales; all other labels, all difficulty bins and all error types are heuristic. Coding difficulty is the tertile of the reference solution's syntax-tree size; science difficulty is the tertile of evidence size plus abstract length.

| | Coding | Science |
|---|---|---|
| Session tasks / transfer variants | 150 / 150 | 150 / 150 |
| Difficulty 1 / 2 / 3 | 50 / 50 / 50 | 50 / 50 / 50 |
| Mean prompt length (characters) | 162 | 2,033 |
| Tasks with operation: understand/plan, execute | 150, 150 | 150, 150 |
| Tasks with operation: debug/verify | 100 | 111 |
| Tasks with operation: explain | 50 | 150 |
| Tests per task (mean) | 3.0 | - |
| SUPPORT / CONTRADICT | - | 75 / 75 |
| Evidence sentences per task (mean) | - | 1.8 |
| Abstract sentences per task (mean) | - | 9.7 |

# Data limitations

**Heuristic annotations.** Operation labels, difficulty bins and error types are heuristic, not human judgements. In coding, operation membership is a deterministic function of difficulty (debug/verify from difficulty 2, explain at difficulty 3), and difficulty bins are tertiles, so equal bin sizes are by construction. We report a 40-task audit comparing two independent annotators with the heuristic labels (see Label-noise audit). Error types are coarse tags derived from solution syntax or claim features (for example, `missed_negation` coincides with the CONTRADICT label) and are not used for any claim in this paper.

**Weak transfer.** Coding transfer variants are near-transfer: same algorithm, new names and numbers. They test retention of the underlying skill, not generalization to a new problem. Science variants match on label and evidence structure, not topic. We therefore interpret transfer as repeated application of the same skill, not far transfer.

**Selection effects.** Requiring an automatic variant removed many candidates (96 of the 263 MBPP problems scanned), which biases the coding pool toward problems with integer inputs. MBPP problems are entry-level and likely appear in the pretraining data of language models, including the model that generates our hints; hints are validated for leakage but memorization cannot be ruled out.

**Scale and scope.** The set has 150 tasks per domain, and every result in this paper is on simulated learners; we make no claim about real humans.