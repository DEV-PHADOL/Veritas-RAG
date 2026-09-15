from db.connection import SessionLocal
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker


def main():
    query = "What is inflation?"

    db = SessionLocal()

    try:
        # Build BM25 index
        hybrid_retriever = HybridRetriever()
        hybrid_retriever.build_index(db)

        # Get hybrid results
        print("\nRunning Hybrid Retrieval...")

        hybrid_results = hybrid_retriever.search(
            db=db,
            query=query,
            top_k=10
        )

        print(f"\nHybrid results: {len(hybrid_results)}")

        # Load reranker
        reranker = Reranker()

        # Rerank results
        print("\nRunning Reranker...")

        reranked_results = reranker.rerank(
            query=query,
            results=hybrid_results,
            top_k=5
        )

        print("\nReranked Results:\n")

        for result in reranked_results:
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Document ID: {result['document_id']}")
            print(f"Page Number: {result['page_number']}")
            print(f"RRF Score: {result['rrf_score']}")
            print(f"Rerank Score: {result['rerank_score']}")
            print(f"Content: {result['content']}")
            print("-" * 60)

    finally:
        db.close()


if __name__ == "__main__":
    main()