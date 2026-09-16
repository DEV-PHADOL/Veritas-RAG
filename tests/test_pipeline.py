from db.connection import SessionLocal
from generation.pipeline import RAGPipeline


def main():

    query = "How much does cybersecurity training cost?"

    print("Initializing RAG pipeline...")

    pipeline = RAGPipeline()

    db = SessionLocal()

    try:

        print("\nProcessing query:")
        print(query)

        result = pipeline.answer(
            db=db,
            query=query,
            top_k=5,
            candidate_k=10,
            document_id=11
        )

        print("\n" + "=" * 60)
        print("GENERATED ANSWER")
        print("=" * 60)

        print(result["answer"])

        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)

        for source in result["sources"]:

            print(
                f"Document ID: {source['document_id']}"
            )

            print(
                f"Chunk ID: {source['chunk_id']}"
            )

            print(
                f"Page Number: {source['page_number']}"
            )

            print("-" * 40)

    finally:

        db.close()


if __name__ == "__main__":
    main()