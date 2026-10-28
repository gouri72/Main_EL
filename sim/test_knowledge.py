
from sim.knowledge import KnowledgeEstimator


def main():
    estimator = KnowledgeEstimator()

    initial = estimator.estimate("coding", "execute")
    assert 0 <= initial["domain_knowledge"] <= 1

    estimator.update("coding", "execute", True, 1)
    estimator.update("coding", "execute", True, 0)
    estimator.update("coding", "execute", False, 2)

    result = estimator.estimate("coding", "execute")
    bins = estimator.knowledge_bins("coding", "execute")

    assert 0 <= result["domain_knowledge"] <= 1
    assert 0 <= result["operation_knowledge"] <= 1
    assert bins["domain_knowledge_bin"] in (0, 1, 2)
    assert bins["operation_knowledge_bin"] in (0, 1, 2)

    print("PASS: observable knowledge estimates")
    print("PASS: knowledge bins")
    print("Estimates:", result)
    print("Bins:", bins)


if __name__ == "__main__":
    main()
