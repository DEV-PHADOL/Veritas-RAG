from generation.llm import GeminiGenerator


class QueryRewriter:

    def __init__(self):
        self.generator = GeminiGenerator()

    def rewrite(self, query: str) -> str:

        prompt = f"""
You are a search query optimization assistant.

Rewrite the following user question into a
clearer search query for document retrieval.

Rules:
1. Preserve the original meaning.
2. Use important keywords.
3. Return only the rewritten query.
4. Do not answer the question.

Original question:
{query}

Rewritten query:
"""

        rewritten_query = self.generator.generate(prompt)

        return rewritten_query.strip()