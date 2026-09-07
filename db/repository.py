from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector
from db.models import Document, DocumentChunk

def insert_document(
    db:Session,
    filename:str,
    chunks:list[dict]
)->Document:
    try:
        document = Document(filename=filename)
        db.add(document)
        db.flush()
        
        for chunk_data in chunks:
            chunk = DocumentChunk(
                content=chunk_data["content"],
                page_number=chunk_data["page_number"],
                embedding=chunk_data["embedding"]
            )
            
            document.chunks.append(chunk)
            
        db.commit()
        
        db.refresh(document)
        
        return document
    
    except Exception as e:
        db.rollback()
        raise e
    

def search_documents(
    db:Session,
    query_embedding:list[float],
    top_k:int=5
):
    distance = DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
    
    results = (
        db.query(DocumentChunk, distance)
        .order_by(distance)
        .limit(top_k)
        .all()
    )
    
    return results


    
    