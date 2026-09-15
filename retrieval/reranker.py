from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        print("Loading reranker model...")

        self.model = CrossEncoder(model_name)

        print("Reranker model loaded.")

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5
    ) -> list[dict]:

        if not results:
            return []

        pairs = [
            (query, result["content"])
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked_results = []

        for result, score in zip(results, scores):
            updated_result = result.copy()
            updated_result["rerank_score"] = float(score)

            reranked_results.append(updated_result)

        reranked_results.sort(
            key=lambda result: result["rerank_score"],
            reverse=True
        )

        return reranked_results[:top_k]