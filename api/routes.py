from fastapi import APIRouter, Depends, HTTPException
from httpcore import request
from httpcore import request
from sqlalchemy.orm import Session

from db.connection import SessionLocal
from generation.pipeline import RAGPipeline
from api.schemas import AskRequest, AskResponse
from generation.llm import GeminiServiceError


router = APIRouter()

pipeline = None

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