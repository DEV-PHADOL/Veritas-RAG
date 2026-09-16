from generation.llm import GeminiGenerator


def main():

    print("Initializing Gemini...")

    generator = GeminiGenerator()

    prompt = """
    Explain cybersecurity awareness training
    in simple language in 3 sentences.
    """

    print("\nSending prompt to Gemini...")

    response = generator.generate(prompt)

    print("\nGemini Response:")
    print(response)


if __name__ == "__main__":
    main()