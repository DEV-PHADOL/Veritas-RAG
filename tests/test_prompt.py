from generation.prompt import build_rag_prompt


def main():

    query = "What is cybersecurity awareness training?"

    context = """
[Source 1]
Document ID: 11
Page Number: 82

Content:
Awareness training helps employees understand
their role in protecting information and systems.
"""

    prompt = build_rag_prompt(
        query=query,
        context=context
    )

    print("\nGenerated Prompt:")
    print("=" * 60)
    print(prompt)
    print("=" * 60)


if __name__ == "__main__":
    main()