
# sim/knowledge.py
"""Observable-history knowledge estimator for HumanFirst-Route."""

from collections import defaultdict


class KnowledgeEstimator:
    """Estimate knowledge from observed attempts, not hidden mastery."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.domain_stats = defaultdict(lambda: {"success": 0, "total": 0})
        self.operation_stats = defaultdict(
            lambda: {"success": 0, "total": 0}
        )

    def update(self, domain, operation, success, level):
        """Update estimates after an observed attempt."""
        if domain not in {"coding", "science"}:
            raise ValueError("domain must be coding or science")
        if not isinstance(success, bool):
            raise ValueError("success must be True or False")
        if not isinstance(level, int) or not 0 <= level <= 4:
            raise ValueError("level must be an integer from 0 to 4")

        for stats in (
            self.domain_stats[domain],
            self.operation_stats[(domain, operation)],
        ):
            stats["total"] += 1
            stats["success"] += int(success)

    @staticmethod
    def _estimate(stats):
        # Beta(2, 2) prior: smooth estimates when data is limited.
        return (stats["success"] + 2) / (stats["total"] + 4)

    def estimate(self, domain, operation):
        """Return domain and operation estimates in [0, 1]."""
        domain_knowledge = self._estimate(self.domain_stats[domain])
        operation_knowledge = self._estimate(
            self.operation_stats[(domain, operation)]
        )

        return {
            "domain_knowledge": round(domain_knowledge, 4),
            "operation_knowledge": round(operation_knowledge, 4),
        }

    def knowledge_bins(self, domain, operation):
        """Convert estimates to low, medium or high knowledge bins."""
        estimates = self.estimate(domain, operation)

        def to_bin(value):
            if value < 0.4:
                return 0
            if value < 0.7:
                return 1
            return 2

        return {
            "domain_knowledge_bin": to_bin(
                estimates["domain_knowledge"]
            ),
            "operation_knowledge_bin": to_bin(
                estimates["operation_knowledge"]
            ),
        }
