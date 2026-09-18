import json

from db.connection import SessionLocal
from generation.pipeline import RAGPipeline
from evals.llm_evaluator import LLMEvaluator

def main():

    with open(
        "evals/dataset.json",
        "r",
        encoding="utf-8"
    ) as file:
        dataset = json.load(file)

    db = SessionLocal()
    pipeline = RAGPipeline()
    evaluator = LLMEvaluator()

    try:

        for item in dataset:

            question = item["question"]

            result = pipeline.answer(
                db=db,
                query=question,
                top_k=5,
                candidate_k=10,
                document_id=item.get("document_id")
            )

            print("\n" + "=" * 70)
            print(f"QUESTION: {question}")
            print("=" * 70)

            print("\nEXPECTED ANSWER:")
            print(item["expected_answer"])

            print("\nGENERATED ANSWER:")
            print(result["answer"])
            
            context = result.get("evaluation_context", "")

            faithfulness = evaluator.evaluate_faithfulness(
                question=question,
                answer=result["answer"],
                context=context
            )

            print("\nFAITHFULNESS:")
            print(faithfulness)

            print("\nRETRIEVAL STATUS:")
            print(
                result.get(
                    "retrieval_status",
                    "relevant"
                )
            )

    finally:
        db.close()


if __name__ == "__main__":
    main()