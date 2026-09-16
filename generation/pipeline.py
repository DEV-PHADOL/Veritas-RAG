from sqlalchemy.orm import Session

from retrieval.hybrid import HybridRetriever
from generation.context_builder import build_context
from generation.prompt import build_rag_prompt
from generation.llm import GeminiGenerator
from retrieval.relevance import RelevanceChecker

class RAGPipeline:

    def __init__(self):

        self.retriever = HybridRetriever()

        self.generator = GeminiGenerator()

        self.relevance_checker = RelevanceChecker(
            threshold=0.0
        )

    def answer(
        self,
        db: Session,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
        document_id: int | None = None
    ) -> dict:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        # 1. Retrieve and rerank
        results = self.retriever.search(
            db=db,
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            document_id=document_id
        )

        # Check retrieval relevance
        is_relevant = self.relevance_checker.is_relevant(
            results
        )

        if not is_relevant:

            return {
                "answer": (
                    "I couldn't find relevant information "
                    "in the provided documents."
                ),
                "sources": [],
                "retrieval_status": "irrelevant"
            }
        # 2. Build context
        context = build_context(results)

        if not context:
            return {
                "answer": (
                    "I don't have enough information "
                    "to answer this question."
                ),
                "sources": []
            }

        # 3. Build prompt
        prompt = build_rag_prompt(
            query=query,
            context=context
        )

        # 4. Generate answer
        answer = self.generator.generate(prompt)

        # 5. Return answer and sources
        sources = [
            {
                "document_id": result["document_id"],
                "chunk_id": result["chunk_id"],
                "page_number": result["page_number"]
            }
            for result in results
        ]

        return {
            "answer": answer,
            "sources": sources
        }