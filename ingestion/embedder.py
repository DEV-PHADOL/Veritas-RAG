import os

from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "gemini-embedding-001"

def embed_chunks(chunks: list[dict])->list[dict]:
    """
        Generate embeddings for multiple chunks and attach
        each embedding to its corresponding chunk.
    """
    
    texts = [chunk["text"] for chunk in chunks]
    
    result = client.models.embed_text(
        model=MODEL_NAME,
        content=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )
    )
    
    for chunk, embedding in zip(chunks, result.embeddings):
        chunk["embedding"] = embedding.values
        
    return chunks