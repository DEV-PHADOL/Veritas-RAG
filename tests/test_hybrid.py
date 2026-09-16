import time

from db.connection import SessionLocal
from retrieval.hybrid import HybridRetriever


def main():

    query = "What is inflation?"

    # ==================================================
    # DATABASE SESSION
    # ==================================================

    print("\nCreating database session...")

    session_start = time.perf_counter()

    db = SessionLocal()

    session_time = (
        time.perf_counter() - session_start
    )

    print(
        f"Session creation time: "
        f"{session_time:.4f}s"
    )

    try:

        # ==================================================
        # CREATE HYBRID RETRIEVER
        # ==================================================

        print("\nCreating Hybrid Retriever...")

        retriever = HybridRetriever()

        # ==================================================
        # BUILD BM25 INDEX
        # ==================================================

        print("\nBuilding BM25 index...")

        bm25_start = time.perf_counter()

        retriever.build_index(db)

        bm25_index_time = (
            time.perf_counter() - bm25_start
        )

        print(
            f"BM25 index build time: "
            f"{bm25_index_time:.4f}s"
        )

        # ==================================================
        # QUERY
        # ==================================================

        print("\nQuery:")
        print(query)

        # ==================================================
        # HYBRID RETRIEVAL
        # ==================================================

        print("\nRunning Hybrid Retrieval...")

        retrieval_start = time.perf_counter()

        results = retriever.search(
            db=db,
            query=query,
            top_k=5,
            candidate_k=10
        )

        total_retrieval_time = (
            time.perf_counter() - retrieval_start
        )

        # ==================================================
        # RESULTS
        # ==================================================

        print("\nHybrid Search Results:\n")

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"Rank: {rank}"
            )

            print(
                f"Chunk ID: "
                f"{result['chunk_id']}"
            )

            print(f"Filename: {result['filename']}")

            print(
                f"Page Number: "
                f"{result['page_number']}"
            )

            print(
                f"RRF Score: "
                f"{result['rrf_score']:.6f}"
            )

            print(
                f"Rerank Score: "
                f"{result['rerank_score']:.6f}"
            )

            print(
                f"Content: "
                f"{result['content']}"
            )

            print("-" * 50)

        # ==================================================
        # PERFORMANCE SUMMARY
        # ==================================================

        print("\n" + "=" * 60)
        print("HYBRID RETRIEVAL PERFORMANCE")
        print("=" * 60)

        print(
            f"Session creation       : "
            f"{session_time:.4f}s"
        )

        print(
            f"BM25 index build       : "
            f"{bm25_index_time:.4f}s"
        )

        print(
            f"Total retrieval        : "
            f"{total_retrieval_time:.4f}s"
        )

        print("=" * 60)

        # ==================================================
        # EXPECTED PIPELINE
        # ==================================================

        print("\nPipeline:")
        print(
            "BM25 + Vector Search "
            "-> RRF -> Reranker -> Top-K"
        )

    finally:

        db.close()


if __name__ == "__main__":
    main()
