from db.connection import SessionLocal
from retrieval.bm25 import BM25Retriever, BM25Retriever


def main():

    db = SessionLocal()

    try:

        # Create BM25 retriever
        retriever = BM25Retriever()

        # Build index from database chunks
        retriever.build_index(db)

        # Test query
        query = "What is inflation?"

        print("\nQuery:")
        print(query)

        # Search
        results = retriever.search(
            query=query,
            top_k=5
        )

        print("\nBM25 Results:\n")

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
                f"Score: "
                f"{result['score']}"
            )

            print(
                f"Content: "
                f"{result['content'][:300]}"
            )

            print("-" * 50)
    finally:

        db.close()


if __name__ == "__main__":

    main()