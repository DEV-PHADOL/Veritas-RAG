from db.connection import SessionLocal
from retrieval.hybrid import HybridRetriever


def main():

    query = "What is inflation?"

    db = SessionLocal()

    try:

        retriever = HybridRetriever()

        print("\nBuilding BM25 index...")
        retriever.build_index(db)

        print("\nQuery:")
        print(query)

        print("\nRunning Hybrid Retrieval...")

        results = retriever.search(
            db=db,
            query=query,
            top_k=5
        )

        print("\nHybrid Search Results:\n")

        for result in results:

            print(
                f"Chunk ID: {result['chunk_id']}"
            )

            print(
                f"Document ID: "
                f"{result['document_id']}"
            )

            print(
                f"Page Number: "
                f"{result['page_number']}"
            )

            print(
                f"RRF Score: "
                f"{result['rrf_score']}"
            )

            print(
                f"Content: "
                f"{result['content']}"
            )

            print("-" * 50)

    finally:

        db.close()


if __name__ == "__main__":
    main()