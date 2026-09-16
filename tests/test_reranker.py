from db.connection import SessionLocal
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker


def main():

    test_queries = [
        "What is cybersecurity awareness training?",
        "What is the population of India?"
    ]

    for query in test_queries:

        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)

        db = SessionLocal()

        try:
            hybrid_retriever = HybridRetriever()

            hybrid_results = hybrid_retriever.search(
                db=db,
                query=query,
                top_k=10
            )

            reranker = Reranker()

            reranked_results = reranker.rerank(
                query=query,
                results=hybrid_results,
                top_k=5
            )

            print("\nReranker Scores:\n")

            for result in reranked_results:

                print(
                    f"Score: {result['rerank_score']:.4f} | "
                    f"Page: {result['page_number']}"
                )

        finally:
            db.close()

if __name__ == "__main__":
    main()