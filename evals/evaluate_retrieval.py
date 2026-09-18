import json

from db.connection import SessionLocal
from retrieval.hybrid import HybridRetriever


def calculate_recall_at_k(results, relevant_ids, k):
    retrieved_ids = {
        result["chunk_id"]
        for result in results[:k]
    }

    return int(
        bool(retrieved_ids.intersection(relevant_ids))
    )


def calculate_mrr(results, relevant_ids):
    for rank, result in enumerate(results, start=1):

        if result["chunk_id"] in relevant_ids:
            return 1 / rank

    return 0.0


def main():

    with open(
        "evals/dataset.json",
        "r",
        encoding="utf-8"
    ) as file:
        dataset = json.load(file)

    db = SessionLocal()
    retriever = HybridRetriever()

    recall_scores = []
    mrr_scores = []

    try:

        for item in dataset:

            question = item["question"]
            relevant_ids = set(
                item["relevant_chunk_ids"]
            )

            results = retriever.search(
                db=db,
                query=question,
                top_k=10,
                candidate_k=20,
                document_id=item.get("document_id")
            )

            recall = calculate_recall_at_k(
                results,
                relevant_ids,
                k=5
            )

            mrr = calculate_mrr(
                results,
                relevant_ids
            )

            recall_scores.append(recall)
            mrr_scores.append(mrr)

            print(f"\nQuestion: {question}")
            print(f"Recall@5: {recall}")
            print(f"MRR: {mrr:.4f}")

        print("\n" + "=" * 50)
        print("FINAL EVALUATION")
        print("=" * 50)

        print(
            f"Average Recall@5: "
            f"{sum(recall_scores) / len(recall_scores):.4f}"
        )

        print(
            f"Mean Reciprocal Rank: "
            f"{sum(mrr_scores) / len(mrr_scores):.4f}"
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()