from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from db.repository import get_chunk_by_content_hash


# 768-dimensional local embedding model
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
OUTPUT_DIMENSION = 768

model = SentenceTransformer(MODEL_NAME)


def get_text_hash(text: str) -> str:
    import hashlib

    normalized_text = text.strip()

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


def embed_chunks(
    chunks: list[dict],
    db: Session | None = None
) -> list[dict]:
    if not chunks:
        return chunks

    texts_to_embed = []
    chunks_to_embed = []

    memory_cache_hits = 0
    database_cache_hits = 0

    # Check existing embeddings first
    for chunk in chunks:

        text = chunk["text"]
        text_hash = get_text_hash(text)

        # In-memory cache
        if text_hash in embedding_cache:
            chunk["embedding"] = embedding_cache[text_hash]
            memory_cache_hits += 1
            continue

        # Database cache
        existing_chunk = None

        if db is not None:
            existing_chunk = get_chunk_by_content_hash(
                db=db,
                content_hash=text_hash
            )

        if existing_chunk is not None:
            chunk["embedding"] = existing_chunk.embedding
            embedding_cache[text_hash] = existing_chunk.embedding
            database_cache_hits += 1
            continue

        texts_to_embed.append(text)
        chunks_to_embed.append((chunk, text_hash))

    print(f"Memory cache hits: {memory_cache_hits}")
    print(f"Database cache hits: {database_cache_hits}")
    print(f"New embeddings required: {len(texts_to_embed)}")

    if not texts_to_embed:
        print("All embeddings found in cache.")
        return chunks

    print(f"Embedding {len(texts_to_embed)} chunks locally...")

    # Generate embeddings locally
    embeddings = model.encode(
        texts_to_embed,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=False,
        show_progress_bar=False
    )

    if len(embeddings) != len(chunks_to_embed):
        raise RuntimeError(
            "Unexpected number of embeddings returned."
        )

    # Attach embeddings to chunks
    for (
        (chunk, text_hash),
        embedding
    ) in zip(chunks_to_embed, embeddings):

        embedding_values = embedding.tolist()

        chunk["embedding"] = embedding_values

        embedding_cache[text_hash] = embedding_values

    return chunks


def embed_query(query: str) -> list[float]:

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=False
    )

    return embedding.tolist()


# In-memory embedding cache
embedding_cache = {}