from generation.query_rewriter import QueryRewriter


def main():

    rewriter = QueryRewriter()

    queries = [
        "What is cybersecurity awareness training?",
        "Why is security training important?",
        "Tell me about training",
        "What are the responsibilities of employees?"
    ]

    for query in queries:

        rewritten_query = rewriter.rewrite(query)

        print("\n" + "=" * 50)
        print("Original Query:")
        print(query)

        print("\nRewritten Query:")
        print(rewritten_query)


if __name__ == "__main__":
    main()