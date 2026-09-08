from sqlalchemy.orm import Session

from db.models import Document, DocumentChunk


def create_document(
    db: Session,
    filename: str
) -> Document:

    document = Document(
        filename=filename
    )

    db.add(document)
    db.commit()

    db.refresh(document)

    return document


def insert_chunks(
    db: Session,
    document: Document,
    chunks: list[dict]
):

    try:

        for chunk_data in chunks:

            chunk = DocumentChunk(
                document_id=document.id,
                content=chunk_data["text"],
                page_number=chunk_data["page_number"],
                embedding=chunk_data["embedding"]
            )

            db.add(chunk)

        db.commit()

    except Exception:
        db.rollback()
        raise


def search_documents(
    db: Session,
    query_embedding: list[float],
    top_k: int = 5
):

    distance = (
        DocumentChunk.embedding
        .cosine_distance(query_embedding)
        .label("distance")
    )

    results = (
        db.query(DocumentChunk, distance)
        .order_by(distance)
        .limit(top_k)
        .all()
    )

    return results