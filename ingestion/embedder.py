import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")


client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-embedding-001"


def embed_chunks(chunks: list[dict]) -> list[dict]:

    batch_size = 10

    for start in range(0, len(chunks), batch_size):

        batch = chunks[start:start + batch_size]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        print(
            f"Embedding chunks "
            f"{start + 1} to {start + len(batch)}..."
        )

        while True:

            try:

                result = client.models.embed_content(
                    model=MODEL_NAME,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=768
                    )
                )

                break

            except ClientError as error:

                if error.code == 429:

                    print(
                        "Rate limit reached. "
                        "Waiting 60 seconds..."
                    )

                    time.sleep(60)

                else:
                    raise

        for chunk, embedding in zip(
            batch,
            result.embeddings
        ):
            chunk["embedding"] = embedding.values

        time.sleep(2)

    return chunks