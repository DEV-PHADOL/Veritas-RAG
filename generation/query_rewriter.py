from generation.llm import GeminiGenerator


class QueryRewriter:

    def __init__(self):
        self.generator = GeminiGenerator()

    def rewrite(self, query: str) -> str:

        prompt = f"""
You are a search query optimization assistant
for an enterprise document retrieval system.

Your task is to rewrite the user's question into
a clear and keyword-rich search query.

Rules:
1. Preserve the exact meaning of the original question.
2. Do not add new topics, assumptions, or information.
3. Keep important technical terms and keywords.
4. Remove unnecessary conversational words.
5. Return only the rewritten search query.
6. Do not answer the question.
7. If the query is already clear, return it with
   minimal changes.
8. Keep the rewritten query concise.

Original question:
{query}

Rewritten search query:
"""

        rewritten_query = self.generator.generate(prompt)

        return rewritten_query.strip()