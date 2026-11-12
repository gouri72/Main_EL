# Dataset provenance and related benchmarks (P3)

This note covers the public data sources used in HumanFirst-Route, **not**
evidence that the proposed assistance policy outperforms either benchmark.

## MBPP — programming

Austin et al. (2021), *Program Synthesis with Large Language Models*,
introduces Mostly Basic Python Problems (MBPP) as a benchmark of short
Python programming tasks. The original benchmark has 974 tasks; each
includes a natural-language prompt, reference program, and automated tests.

- Paper: https://arxiv.org/abs/2108.07732
- Original dataset repository: https://github.com/google-research/google-research/tree/master/mbpp

**Our use:** filter suitable base problems, preserve executable reference
solutions, label reasoning operations using AST heuristics, and generate
paired *near-transfer* variants by renaming functions and modifying integer
test arguments. The new variants are not independent human-authored
problems. Our outcome metric is simulated unaided learner performance, not
MBPP's usual code-generation accuracy.

## SciFact — scientific evidence verification

Wadden et al. (2020), *Fact or Fiction: Verifying Scientific Claims*,
introduces expert-written claims linked to abstracts, evidence rationales,
and SUPPORT/REFUTE labels.

- Paper: https://aclanthology.org/2020.emnlp-main.609/
- Original dataset repository: https://github.com/allenai/scifact

**Our use:** select eligible annotated claims from available train/dev
records, retain the source label/evidence, and pair each base claim with
another same-label, same-evidence-bucket claim from a distinct abstract.
For our task schema the negative label is spelled `CONTRADICT`.
The `execute` operation is marked gold-grounded because locating annotated
evidence is intrinsic to the SciFact task; the other operation labels and
difficulty scores are heuristic.

## Why these are appropriate, and their limitations

MBPP supplies executable, testable short programming problems; SciFact
supplies rationale-grounded scientific verification problems. Together
they introduce distinct task-operation semantics while enabling
reproducible task construction from public sources.

Neither original benchmark validates our custom assistance-level policy.
Coding transfer pairs largely measure the same algorithm under changed
inputs; scientific transfer pairs may require different domain knowledge.
The simulation models transfer through latent mastery and not by grading
real learner responses to these pairs. We explicitly avoid treating our
simulator's transfer percentage as benchmark accuracy or real student
learning improvement.

Original references:

1. Austin, J., et al. (2021). *Program Synthesis with Large Language Models*.
   arXiv:2108.07732. https://arxiv.org/abs/2108.07732
2. Wadden, D., et al. (2020). *Fact or Fiction: Verifying Scientific Claims*.
   EMNLP, 7534–7550. DOI: 10.18653/v1/2020.emnlp-main.609.
