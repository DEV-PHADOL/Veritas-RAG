import time
from collections import defaultdict

from sqlalchemy.orm import Session

from ingestion.embedder import embed_query
from retrieval.bm25 import BM25Retriever
from retrieval.reranker import Reranker
from retrieval.vector import VectorRetriever


class HybridRetriever:

    def __init__(self):
        # BM25 retriever
        self.bm25_retriever = BM25Retriever()

        # Vector retriever
        self.vector_retriever = VectorRetriever()

        # Cross-encoder reranker
        self.reranker = Reranker()

        # Track whether BM25 index has been built
        self.bm25_ready = False

        # Track which document BM25 is indexed for
        self.bm25_document_id = None

    def build_index(
        self,
        db: Session,
        document_id: int | None = None
    ):
        """
        Build the BM25 index.
        """

        if (
            self.bm25_ready
            and self.bm25_document_id == document_id
        ):
            return

        self.bm25_retriever.build_index(
            db=db,
            document_id=document_id
        )

        self.bm25_ready = True
        self.bm25_document_id = document_id

    def invalidate_index(self):
        self.bm25_ready = False
        self.bm25_document_id = None
    
    def search(
        self,
        db: Session,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
        rrf_k: int = 60,
        document_id: int | None = None
    ) -> list[dict]:

        # ==================================================
        # START TOTAL TIMER
        # ==================================================

        total_start = time.perf_counter()

        # ==================================================
        # 1. BUILD / CHECK BM25 INDEX
        # ==================================================

        bm25_index_start = time.perf_counter()

        self.build_index(
            db=db,
            document_id=document_id
        )

        bm25_index_time = (
            time.perf_counter() - bm25_index_start
        )

        # ==================================================
        # 2. BM25 SEARCH
        # ==================================================

        bm25_start = time.perf_counter()

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=candidate_k
        )

        bm25_time = time.perf_counter() - bm25_start

        # ==================================================
        # 3. QUERY EMBEDDING
        # ==================================================

        embedding_start = time.perf_counter()

        query_embedding = embed_query(query)

        embedding_time = (
            time.perf_counter() - embedding_start
        )

        # ==================================================
        # 4. VECTOR SEARCH
        # ==================================================

        vector_start = time.perf_counter()

        vector_results = self.vector_retriever.search(
            db=db,
            query_embedding=query_embedding,
            top_k=candidate_k,
            document_id=document_id
        )

        vector_time = (
            time.perf_counter() - vector_start
        )

        # ==================================================
        # 5. RRF
        # ==================================================

        rrf_start = time.perf_counter()

        rrf_scores = defaultdict(float)
        results_by_id = {}

        # BM25 results
        for rank, result in enumerate(
            bm25_results,
            start=1
        ):
            chunk_id = result["chunk_id"]

            rrf_scores[chunk_id] += (
                1 / (rrf_k + rank)
            )

            results_by_id[chunk_id] = result.copy()

        # Vector results
        for rank, result in enumerate(
            vector_results,
            start=1
        ):
            chunk_id = result["chunk_id"]

            rrf_scores[chunk_id] += (
                1 / (rrf_k + rank)
            )

            if chunk_id not in results_by_id:
                results_by_id[chunk_id] = result.copy()

        ranked_results = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        rrf_results = []

        for chunk_id, score in ranked_results[:candidate_k]:

            result = results_by_id[chunk_id].copy()

            result["rrf_score"] = score

            rrf_results.append(result)

        rrf_time = time.perf_counter() - rrf_start

        # ==================================================
        # 6. RERANKING
        # ==================================================

        reranker_start = time.perf_counter()

        reranked_results = self.reranker.rerank(
            query=query,
            results=rrf_results,
            top_k=top_k
        )

        reranker_time = (
            time.perf_counter() - reranker_start
        )

        # ==================================================
        # TOTAL TIME
        # ==================================================

        total_time = (
            time.perf_counter() - total_start
        )

        # ==================================================
        # PRINT TIMING
        # ==================================================

        print("\n" + "=" * 50)
        print("RETRIEVAL TIMING")
        print("=" * 50)

        print(
            f"BM25 index/check : {bm25_index_time:.4f}s"
        )

        print(
            f"BM25 search      : {bm25_time:.4f}s"
        )

        print(
            f"Query embedding  : {embedding_time:.4f}s"
        )

        print(
            f"Vector search    : {vector_time:.4f}s"
        )

        print(
            f"RRF              : {rrf_time:.4f}s"
        )

        print(
            f"Reranker         : {reranker_time:.4f}s"
        )

        print("-" * 50)

        print(
            f"TOTAL RETRIEVAL  : {total_time:.4f}s"
        )

        print("=" * 50)

        return reranked_results