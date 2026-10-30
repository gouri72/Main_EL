
from sim.knowledge import KnowledgeEstimator


def main():
    estimator = KnowledgeEstimator()

    # Initial estimates should be valid probabilities.
    initial = estimator.estimate("coding", "execute")
    assert 0.0 <= initial["domain_knowledge"] <= 1.0
    assert 0.0 <= initial["operation_knowledge"] <= 1.0

    # Check observable history updates and knowledge bins.
    estimator.update("coding", "execute", True, 1)
    estimator.update("coding", "execute", True, 0)
    estimator.update("coding", "execute", False, 2)

    result = estimator.estimate("coding", "execute")
    bins = estimator.knowledge_bins("coding", "execute")

    assert all(0.0 <= value <= 1.0 for value in result.values())
    assert bins["domain_knowledge_bin"] in (0, 1, 2)
    assert bins["operation_knowledge_bin"] in (0, 1, 2)

    # Independent success should provide stronger evidence than
    # success after receiving a full solution.
    independent = KnowledgeEstimator()
    full_solution = KnowledgeEstimator()

    independent.update("coding", "execute", True, 0)
    full_solution.update("coding", "execute", True, 4)

    independent_estimate = independent.estimate("coding", "execute")
    assisted_estimate = full_solution.estimate("coding", "execute")

    assert (
        independent_estimate["operation_knowledge"]
        > assisted_estimate["operation_knowledge"]
    )

    # Reset must clear prior observations.
    estimator.reset()
    reset_estimate = estimator.estimate("coding", "execute")
    assert reset_estimate == initial

    print("PASS: observable knowledge estimates")
    print("PASS: knowledge bins")
    print("PASS: assistance-aware evidence weighting")
    print("PASS: estimator reset")
    print("Estimates after observations:", result)
    print("Independent success:", independent_estimate)
    print("Full-solution success:", assisted_estimate)


if __name__ == "__main__":
    main()
