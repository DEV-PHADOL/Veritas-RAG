import hashlib

from sqlalchemy.orm import Session

from db.models import Document, DocumentChunk

def calculate_content_hash(text: str) -> str:
    return hashlib.sha256(
        text.strip().encode("utf-8")
    ).hexdigest()

# ==========================================
# Create Document
# ==========================================

def create_document(
    db: Session,
    filename: str,
    file_hash: str
) -> Document:

    document = Document(

        filename=filename,

        file_hash=file_hash,

        ingestion_status="PROCESSING"

    )

    db.add(document)

    db.commit()

    db.refresh(document)

    return document


# ==========================================
# Insert Chunks
# ==========================================

def insert_chunks(
    db: Session,
    document: Document,
    chunks: list[dict]
):

    try:

        for chunk_data in chunks:

            content = chunk_data["text"]

            content_hash = calculate_content_hash(
                content
            )

            chunk = DocumentChunk(

                document_id=document.id,

                content=content,

                content_hash=content_hash,

                page_number=(
                    chunk_data["page_number"]
                ),

                embedding=(
                    chunk_data["embedding"]
                )

            )

            db.add(chunk)

        db.commit()

    except Exception:

        db.rollback()

        raise

# ==========================================
# Get Document By Hash
# ==========================================

def get_document_by_hash(
    db: Session,
    file_hash: str
):

    return (

        db.query(Document)

        .filter(
            Document.file_hash == file_hash
        )

        .first()

    )


# ==========================================
# Update Ingestion Status
# ==========================================

def update_document_status(

    db: Session,

    document: Document,

    status: str

):

    document.ingestion_status = status

    db.commit()

    db.refresh(document)

    return document


# ==========================================
# Mark Document As Completed
# ==========================================

def mark_document_completed(

    db: Session,

    document: Document

):

    return update_document_status(

        db=db,

        document=document,

        status="COMPLETED"

    )


# ==========================================
# Mark Document As Failed
# ==========================================

def mark_document_failed(db: Session, document: Document):
    try:
        # Remove any chunks that were inserted before ingestion failed.
        (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_id == document.id
            )
            .delete(synchronize_session=False)
        )

        # Mark the document as failed.
        document.ingestion_status = "FAILED"

        db.commit()

    except Exception:
        db.rollback()
        raise


# ==========================================
# Search Documents
# ==========================================

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

        db.query(

            DocumentChunk,

            distance

        )

        .order_by(distance)

        .limit(top_k)

        .all()

    )


    return results


def delete_document_chunks(db: Session, document_id: int):
    try:
        (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .delete(synchronize_session=False)
        )

        db.commit()

    except Exception:
        db.rollback()
        raise
    
def get_chunk_by_content_hash(
    db: Session,
    content_hash: str
):
    return (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.content_hash == content_hash
        )
        .first()
    )