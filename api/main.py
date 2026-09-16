from fastapi import FastAPI

from api.routes import router


app = FastAPI(
    title="VERITAS-RAG API",
    description="Hybrid RAG with Reranking and Corrective RAG",
    version="1.0.0"
)


app.include_router(router)