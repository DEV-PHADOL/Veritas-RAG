from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.connection import SessionLocal
from generation.pipeline import RAGPipeline
from api.schemas import AskRequest, AskResponse
from generation.llm import GeminiServiceError

import os
import tempfile

from fastapi import UploadFile, File
from Scripts.ingest import ingest_pdf


router = APIRouter()

pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:

    global pipeline

    if pipeline is None:

        print("Initializing RAG pipeline...")

        pipeline = RAGPipeline()

    return pipeline

def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


@router.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    db: Session = Depends(get_db)
):

    try:

        rag_pipeline = get_pipeline()

        result = rag_pipeline.answer(
            db=db,
            query=request.query,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            document_id=request.document_id
        )

        return result

    except GeminiServiceError as error:

        print(f"Gemini Service Error: {error}")

        raise HTTPException(
            status_code=503,
            detail=(
                "The AI service is temporarily unavailable. "
                "Please try again later."
            )
        )

    except Exception as error:

        print(f"API Error: {error}")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question."
        )
        
@router.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "VERITAS-RAG API"
    }
    
@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    temp_path = None

    try:

        file_content = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(file_content)
            temp_path = temp_file.name

        document = ingest_pdf(temp_path)
        if pipeline is not None:
            pipeline.retriever.invalidate_index()

        return {
            "message": "Document ingested successfully",
            "document_id": document.id,
            "filename": document.filename,
            "status": document.ingestion_status
        }

    except Exception as error:

        print(f"Ingestion API Error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed."
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)