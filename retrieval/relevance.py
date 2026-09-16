class RelevanceChecker:

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def is_relevant(
        self,
        results: list[dict]
    ) -> bool:

        if not results:
            return False

        best_score = max(
            result["rerank_score"]
            for result in results
        )

        return best_score >= self.threshold