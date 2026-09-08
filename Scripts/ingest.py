from pathlib import Path

from ingestion.loader import load_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from ingestion.embedder import embed_chunks

from db.connection import SessionLocal
from db.repository import create_document, insert_chunks


BATCH_SIZE = 10


def ingest_pdf(file_path: str):

    # 1. Load PDF
    print("Loading PDF...")

    pages = load_pdf(file_path)

    # 2. Clean text
    print("Cleaning text...")

    cleaned_pages = []

    for page in pages:

        cleaned_text = clean_text(
            page["text"]
        )

        if cleaned_text:

            cleaned_pages.append({
                "page_number": page["page_number"],
                "text": cleaned_text
            })

    # 3. Create chunks
    print("Creating chunks...")

    source = Path(file_path).name

    chunks = chunk_text(
        cleaned_pages,
        source
    )

    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        print("No chunks created.")
        return

    # 4. Create database session
    db = SessionLocal()

    try:

        # 5. Create document once
        print("Creating document...")

        document = create_document(
            db=db,
            filename=source
        )

        print(
            f"Document created successfully. "
            f"Document ID: {document.id}"
        )

        # 6. Process chunks in batches
        total_chunks = len(chunks)

        for start in range(
            0,
            total_chunks,
            BATCH_SIZE
        ):

            batch = chunks[
                start:start + BATCH_SIZE
            ]

            batch_number = (
                start // BATCH_SIZE
            ) + 1

            print(
                f"\nProcessing batch {batch_number}"
            )

            print(
                f"Chunks "
                f"{start + 1} to "
                f"{start + len(batch)}"
            )

            # Generate embeddings only for this batch
            print("Generating embeddings...")

            embedded_batch = embed_chunks(
                batch
            )

            print("Saving chunks to database...")

            # Save this batch
            insert_chunks(
                db=db,
                document=document,
                chunks=embedded_batch
            )

            print(
                f"Batch {batch_number} completed."
            )

        print("\nIngestion completed successfully!")

        print(
            f"Document ID: {document.id}"
        )

        print(
            f"Total chunks saved: {total_chunks}"
        )

    except Exception as e:

        print(
            f"\nIngestion failed: {str(e)}"
        )

        raise

    finally:

        db.close()


if __name__ == "__main__":

    pdf_path = (
        "data/raw/"
        "rbi_annual_report_2024_25.pdf"
    )

    ingest_pdf(pdf_path)