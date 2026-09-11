import os
import time
import hashlib

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError


load_dotenv()


# ==========================================
# Configuration
# ==========================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-embedding-001"

OUTPUT_DIMENSION = 768

# Maximum chunks per embedding request
BATCH_SIZE = 10

# Retry configuration
INITIAL_RETRY_DELAY = 5
MAX_RETRY_DELAY = 60


# ==========================================
# Gemini Client
# ==========================================

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set."
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ==========================================
# In-Memory Embedding Cache
# ==========================================

embedding_cache = {}


def get_text_hash(text: str) -> str:

    normalized_text = text.strip()

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


# ==========================================
# Generate Embeddings
# ==========================================

def embed_chunks(
    chunks: list[dict]
) -> list[dict]:

    if not chunks:
        return chunks


    texts_to_embed = []
    chunks_to_embed = []


    # ==========================================
    # Check Cache
    # ==========================================

    for chunk in chunks:

        text = chunk["text"]

        text_hash = get_text_hash(text)


        if text_hash in embedding_cache:

            chunk["embedding"] = (
                embedding_cache[text_hash]
            )

        else:

            texts_to_embed.append(text)

            chunks_to_embed.append(
                (chunk, text_hash)
            )


    # ==========================================
    # All Found In Cache
    # ==========================================

    if not texts_to_embed:

        print(
            "All embeddings found in cache."
        )

        return chunks


    print(
        f"Embedding {len(texts_to_embed)} chunks..."
    )


    # ==========================================
    # Retry API Request
    # ==========================================

    retry_delay = INITIAL_RETRY_DELAY
    retry_count = 0


    while True:

        try:

            result = (
                client.models.embed_content(

                    model=MODEL_NAME,

                    contents=texts_to_embed,

                    config=types.EmbedContentConfig(

                        task_type=(
                            "RETRIEVAL_DOCUMENT"
                        ),

                        output_dimensionality=(
                            OUTPUT_DIMENSION
                        )

                    )

                )
            )


            # API request successful
            break


        except ClientError as error:

            if error.code == 429:

                retry_count += 1


                print(

                    f"Rate limit reached. "
                    f"Retry attempt {retry_count}. "
                    f"Waiting {retry_delay} seconds..."

                )


                time.sleep(
                    retry_delay
                )


                # Exponential backoff
                retry_delay = min(

                    retry_delay * 2,

                    MAX_RETRY_DELAY

                )


            else:

                raise


    # ==========================================
    # Attach Embeddings
    # ==========================================

    for (
        (chunk, text_hash),
        embedding
    ) in zip(

        chunks_to_embed,

        result.embeddings

    ):


        embedding_values = (
            embedding.values
        )


        # Attach embedding

        chunk["embedding"] = (
            embedding_values
        )


        # Save in memory cache

        embedding_cache[
            text_hash
        ] = embedding_values


    return chunks


def embed_query(
    query:str
) -> list[float]:
    
    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )
        
    result = client.models.embed_content(
        model=MODEL_NAME,
        contents=[query],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=(OUTPUT_DIMENSION)
        )
    )
    
    return result.embeddings[0].values