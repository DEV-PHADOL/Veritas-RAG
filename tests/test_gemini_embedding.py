from ingestion.embedder import embed_chunks


def test_embedding():

    chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "The Reserve Bank of India is the central bank of India.",
            "page_number": 1,
            "source": "test.pdf"
        }
    ]

    result = embed_chunks(chunks)

    print("Embedding dimension:", len(result[0]["embedding"]))
    print("First 5 values:", result[0]["embedding"][:5])

    assert "embedding" in result[0]
    assert len(result[0]["embedding"]) == 768