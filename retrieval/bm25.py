import re

from rank_bm25 import BM25Okapi

from db.models import DocumentChunk


class BM25Retriever:

    def __init__(self):

        self.bm25 = None
        self.chunks = []


    def tokenize(self, text: str) -> list[str]:

        return re.findall(
            r"\b\w+\b",
            text.lower()
        )


    def build_index(self, db):

        print("Building BM25 index...")

        chunks = (
            db.query(DocumentChunk)
            .all()
        )

        if not chunks:

            raise ValueError(
                "No document chunks found."
            )


        tokenized_chunks = []

        self.chunks = []


        for chunk in chunks:

            tokens = self.tokenize(
                chunk.content
            )

            tokenized_chunks.append(
                tokens
            )

            self.chunks.append({

                "chunk_id": chunk.id,

                "document_id": chunk.document_id,

                "content": chunk.content,

                "page_number": chunk.page_number

            })


        self.bm25 = BM25Okapi(
            tokenized_chunks
        )


        print(
            f"BM25 index built for "
            f"{len(chunks)} document chunks."
        )


    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:

        if self.bm25 is None:

            raise ValueError(
                "BM25 index is not built."
            )


        tokenized_query = self.tokenize(
            query
        )


        scores = self.bm25.get_scores(
            tokenized_query
        )


        ranked_indices = sorted(

            range(len(scores)),

            key=lambda i: scores[i],

            reverse=True

        )[:top_k]


        results = []


        for index in ranked_indices:

            chunk = self.chunks[index]


            results.append({

                "chunk_id": chunk["chunk_id"],

                "document_id": (
                    chunk["document_id"]
                ),

                "content": (
                    chunk["content"]
                ),

                "page_number": (
                    chunk["page_number"]
                ),

                "score": float(
                    scores[index]
                )

            })


        return results