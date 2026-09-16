from generation.query_rewriter import QueryRewriter


def main():

    rewriter = QueryRewriter()

    query = "Tell me about training"

    rewritten_query = rewriter.rewrite(query)

    print("\nOriginal Query:")
    print(query)

    print("\nRewritten Query:")
    print(rewritten_query)


if __name__ == "__main__":
    main()