from retrieval.relevance import RelevanceChecker


def main():

    checker = RelevanceChecker(threshold=0.0)

    relevant_results = [
        {"rerank_score": 8.6183},
        {"rerank_score": 5.5949},
        {"rerank_score": 4.5752}
    ]

    irrelevant_results = [
        {"rerank_score": -9.2559},
        {"rerank_score": -11.0947},
        {"rerank_score": -11.2587}
    ]

    print("Relevant query:")
    print(
        checker.is_relevant(relevant_results)
    )

    print("\nIrrelevant query:")
    print(
        checker.is_relevant(irrelevant_results)
    )


if __name__ == "__main__":
    main()