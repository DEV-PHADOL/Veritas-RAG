from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(pages: list[dict], source: str) -> list[dict]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            ""
        ]
    )

    chunks = []
    chunk_counter = 1

    for page in pages:

        text = page["text"].strip()

        if not text:
            continue

        page_chunks = splitter.split_text(text)

        for chunk in page_chunks:

            chunks.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "text": chunk,
                "page_number": page["page_number"],
                "source": source
            })

            chunk_counter += 1

    return chunks