from pydantic import BaseModel, Field


class AskRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="User question"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10
    )

    candidate_k: int = Field(
        default=10,
        ge=5,
        le=50
    )

    document_id: int | None = Field(
        default=None,
        ge=1
    )
    
class SourceResponse(BaseModel):

    document_id: int
    chunk_id: int
    page_number: int


class AskResponse(BaseModel):

    answer: str

    sources: list[SourceResponse]

    retrieval_status: str = "relevant"