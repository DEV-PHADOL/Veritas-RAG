def build_rag_prompt(
    query: str,
    context: str
) -> str:

    prompt = f"""
You are a helpful enterprise knowledge assistant.

Answer the user's question using ONLY the
provided context.

Rules:
1. Do not use outside knowledge.
2. Do not make up information.
3. If the context does not contain enough information,
   say:
   "I don't have enough information to answer
   this question based on the provided documents."
4. Give a clear and concise answer.
5. Cite the page number from the provided context
   when making factual claims.
6. Never invent page numbers or source references.
7. If multiple sources support your answer,
   cite the relevant pages.
8. Do not mention Source 1, Source 2, etc.
   Use document and page information instead.

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    return prompt