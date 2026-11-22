## 6. Reproducibility

### Robustness experiment

```bash
python -m sim.robustness
python sim/plot_robustness.py
```

### Cross-family experiment

```bash
python -m sim.cross_family
python sim/analyze_cross_family.py
```

### Tests

The Person 2 validation suite includes tests for:

- Environment behavior
- Assistance probability mechanisms
- Observation construction
- Reproducibility

The dedicated Person 2 tests and broader regression checks were run during development.

## 7. Limitations

- The experiments use simulated learners rather than real student participants.
- Transfer is modeled using a simulated mastery-based proxy.
- Simulator families encode predefined behavioral assumptions.
- Confidence intervals reflect episode-level variability, not training-seed variability.
- The training budget and evaluation setup are preliminary.

## 8. Summary

Person 2 implemented and evaluated simulator mechanisms, conducted robustness experiments, ran cross-family frozen-policy evaluation, generated result summaries and figures, and validated the implementation through dedicated tests and regression checks.
