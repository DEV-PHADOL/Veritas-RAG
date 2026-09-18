from sqlalchemy.orm import Session

from retrieval.hybrid import HybridRetriever
from generation.context_builder import build_context
from generation.prompt import build_rag_prompt
from generation.llm import GeminiGenerator
from retrieval.relevance import RelevanceChecker
from generation.query_rewriter import QueryRewriter

class RAGPipeline:

    def __init__(self):

        self.retriever = HybridRetriever()

        self.generator = GeminiGenerator()

        self.relevance_checker = RelevanceChecker(
            threshold=0.0
        )
        
        self.query_rewriter = QueryRewriter()

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
        
        # 1. Rewrite the user query
        try:

            rewritten_query = self.query_rewriter.rewrite(query)

        except Exception as error:

            print(
                f"Query rewriting failed: {error}"
            )

            print(
                "Falling back to original query."
            )

            rewritten_query = query

        print("\nOriginal Query:")
        print(query)

        print("\nRewritten Query:")
        print(rewritten_query)

        
        results = self.retriever.search(
            db=db,
            query=rewritten_query,
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
                "sources": [],
                "retrieval_status": "irrelevant"
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
                "filename": result["filename"],
                "chunk_id": result["chunk_id"],
                "page_number": result["page_number"],
                "content": result["content"][:300]
            }
            for result in results
        ]

        return {
            "answer": answer,
            "sources": sources,
            "evaluation_context": context,
            "retrieval_status": "relevant"
        }