from sqlalchemy.orm import Session

from db.models import DocumentChunk


class VectorRetriever:

    def search(
        self,
        db: Session,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: int | None = None
    ) -> list[dict]:

        distance = (
            DocumentChunk.embedding
            .cosine_distance(query_embedding)
            .label("distance")
        )

        query = (
            db.query(
                DocumentChunk,
                distance
            )
        )

        # Filter by document when document_id is provided
        if document_id is not None:
            query = query.filter(
                DocumentChunk.document_id == document_id
            )

        results = (
            query
            .order_by(distance)
            .limit(top_k)
            .all()
        )

        formatted_results = []

        for chunk, distance_value in results:
            formatted_results.append({
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "page_number": chunk.page_number,
                "score": float(distance_value)
            })

        return formatted_results