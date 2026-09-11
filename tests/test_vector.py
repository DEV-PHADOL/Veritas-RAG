from db.connection import SessionLocal
from ingestion.embedder import embed_query
from retrieval.vector import VectorRetriever


def main():

    db = SessionLocal()

    try:

        # ==========================================
        # 1. User Query
        # ==========================================

        query = "What is inflation?"

        print("\nQuery:")
        print(query)


        # ==========================================
        # 2. Generate Query Embedding
        # ==========================================

        print("\nGenerating query embedding...")

        query_embedding = embed_query(
            query
        )


        # ==========================================
        # 3. Vector Search
        # ==========================================

        print("Searching vector database...")

        retriever = VectorRetriever()

        results = retriever.search(
            db=db,
            query_embedding=query_embedding,
            top_k=5
        )


        # ==========================================
        # 4. Print Results
        # ==========================================

        print("\nVector Search Results:\n")

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
                f"Distance: "
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