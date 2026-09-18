import json

from db.connection import SessionLocal
from retrieval.hybrid import HybridRetriever


def main():

    with open("evals/dataset.json", "r", encoding="utf-8") as file:
        dataset = json.load(file)

    db = SessionLocal()

    try:

        retriever = HybridRetriever()

        for item in dataset:

            question = item["question"]
            document_id = item.get("document_id")

            print("\n" + "=" * 70)
            print(f"QUESTION: {question}")
            print("=" * 70)

            results = retriever.search(
                db=db,
                query=question,
                top_k=10,
                candidate_k=20,
                document_id=document_id
            )

            for rank, result in enumerate(results, start=1):

                print(
                    f"\nRank: {rank}"
                    f"\nChunk ID: {result['chunk_id']}"
                    f"\nPage: {result['page_number']}"
                    f"\nRerank Score: {result.get('rerank_score')}"
                    f"\nContent: {result['content'][:300]}"
                )

    finally:

        db.close()


if __name__ == "__main__":
    main()