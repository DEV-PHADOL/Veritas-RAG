from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.connection import SessionLocal
from generation.pipeline import RAGPipeline
from api.schemas import AskRequest, AskResponse


router = APIRouter()

pipeline = RAGPipeline()


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

        result = pipeline.answer(
            db=db,
            query=request.query,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            document_id=request.document_id
        )

        return result

    except Exception as error:

        print(f"API Error: {error}")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question."
        )