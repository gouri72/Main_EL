
"""Observable-history knowledge estimator for HumanFirst-Route."""

from collections import defaultdict


class KnowledgeEstimator:
    """Estimate knowledge from observed outcomes, never hidden learner state."""

    # Evidence weights: more assistance means weaker evidence of independent
    # knowledge. These are modeling assumptions, not empirically fitted values.
    LEVEL_WEIGHTS = {
        0: 1.00,
        1: 0.80,
        2: 0.60,
        3: 0.35,
        4: 0.15,
    }

    def __init__(self):
        self.reset()

    def reset(self):
        self.domain_stats = defaultdict(self._new_stats)
        self.operation_stats = defaultdict(self._new_stats)

    @staticmethod
    def _new_stats():
        return {
            "weighted_success": 0.0,
            "evidence": 0.0,
            "attempts": 0,
        }

    def update(self, domain, operation, success, level):
        """Update estimates using an observed attempt."""
        if domain not in {"coding", "science"}:
            raise ValueError("domain must be coding or science")
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")
        if not isinstance(success, bool):
            raise ValueError("success must be True or False")
        if type(level) is not int or level not in self.LEVEL_WEIGHTS:
            raise ValueError("level must be an integer from 0 to 4")

        weight = self.LEVEL_WEIGHTS[level]

        for stats in (
            self.domain_stats[domain],
            self.operation_stats[(domain, operation)],
        ):
            stats["attempts"] += 1
            stats["evidence"] += weight
            stats["weighted_success"] += weight * int(success)

    @staticmethod
    def _estimate(stats):
        # Beta(2, 2) prior with fractional, assistance-weighted evidence.
        return (stats["weighted_success"] + 2.0) / (
            stats["evidence"] + 4.0
        )

    def estimate(self, domain, operation):
        """Return domain and operation estimates between 0 and 1."""
        if domain not in {"coding", "science"}:
            raise ValueError("domain must be coding or science")
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")

        domain_knowledge = self._estimate(self.domain_stats[domain])
        operation_knowledge = self._estimate(
            self.operation_stats[(domain, operation)]
        )

        return {
            "domain_knowledge": round(domain_knowledge, 4),
            "operation_knowledge": round(operation_knowledge, 4),
        }

    def knowledge_bins(self, domain, operation):
        """Convert estimates to low, medium, or high knowledge bins."""
        estimates = self.estimate(domain, operation)

        def to_bin(value):
            if value < 0.4:
                return 0
            if value < 0.7:
                return 1
            return 2

        return {
            "domain_knowledge_bin": to_bin(estimates["domain_knowledge"]),
            "operation_knowledge_bin": to_bin(
                estimates["operation_knowledge"]
            ),
        }
