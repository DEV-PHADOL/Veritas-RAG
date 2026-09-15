from collections import defaultdict

from sqlalchemy.orm import Session

from retrieval.bm25 import BM25Retriever
from retrieval.vector import VectorRetriever
from ingestion.embedder import embed_query


class HybridRetriever:

    def __init__(self):
        self.bm25_retriever = BM25Retriever()
        self.vector_retriever = VectorRetriever()
        self.bm25_ready = False

    def build_index(self, db: Session):
        """
        Build BM25 index once.
        """
        if not self.bm25_ready:
            self.bm25_retriever.build_index(db)
            self.bm25_ready = True

    def search(
        self,
        db: Session,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60
    ) -> list[dict]:

        # 1. BM25 Search
        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=top_k
        )

        # 2. Generate query embedding
        query_embedding = embed_query(query)

        # 3. Vector Search
        vector_results = self.vector_retriever.search(
            db=db,
            query_embedding=query_embedding,
            top_k=top_k
        )

        # 4. Calculate RRF scores
        rrf_scores = defaultdict(float)
        results_by_id = {}

        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result["chunk_id"]

            rrf_scores[chunk_id] += 1 / (rrf_k + rank)
            results_by_id[chunk_id] = result

        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result["chunk_id"]

            rrf_scores[chunk_id] += 1 / (rrf_k + rank)
            results_by_id[chunk_id] = result

        # 5. Sort by RRF score
        ranked_results = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # 6. Return final results
        final_results = []

        for chunk_id, score in ranked_results[:top_k]:
            result = results_by_id[chunk_id].copy()
            result["rrf_score"] = score

            final_results.append(result)

        return final_results