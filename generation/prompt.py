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
   say: "I don't have enough information to answer
   this question based on the provided documents."
4. Give a clear and concise answer.
5. Include source references when relevant.

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    return prompt