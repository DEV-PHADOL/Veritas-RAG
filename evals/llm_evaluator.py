from generation.llm import GeminiGenerator


class LLMEvaluator:

    def __init__(self):
        self.generator = GeminiGenerator()

    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        context: str
    ) -> str:

        prompt = f"""
You are evaluating a RAG system.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Task:
Determine whether the generated answer is
fully supported by the retrieved context.

Return ONLY one label:
FAITHFUL
or
NOT_FAITHFUL

Do not provide explanations.
"""

        result = self.generator.generate(prompt)

        return result.strip()