from pathlib import Path
import hashlib

from ingestion.loader import load_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from ingestion.embedder import embed_chunks

from db.connection import SessionLocal

from db.repository import (
    create_document,
    insert_chunks,
    get_document_by_hash,
    mark_document_completed,
    mark_document_failed
)


# ==========================================
# Configuration
# ==========================================

INGESTION_BATCH_SIZE = 10


# ==========================================
# Calculate File Hash
# ==========================================

def calculate_file_hash(
    file_path: str
) -> str:

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ==========================================
# Ingest PDF
# ==========================================

def ingest_pdf(file_path: str):

    source = Path(file_path).name

    print("Checking document...")

    file_hash = calculate_file_hash(
        file_path
    )

    db = SessionLocal()

    document = None

    try:

        # ==========================================
        # Check Existing Document
        # ==========================================

        existing_document = get_document_by_hash(
            db=db,
            file_hash=file_hash
        )


        # ------------------------------------------
        # Document Already Successfully Ingested
        # ------------------------------------------

        if (
            existing_document
            and existing_document.ingestion_status
            == "COMPLETED"
        ):

            print(
                "Document already fully ingested."
            )

            print(
                f"Document ID: "
                f"{existing_document.id}"
            )

            print(
                "Skipping ingestion."
            )

            return existing_document


        # ------------------------------------------
        # Failed / Incomplete Document
        # ------------------------------------------

        if existing_document:

            print(
                "Incomplete document found."
            )

            print(
                f"Document ID: "
                f"{existing_document.id}"
            )

            print(
                f"Previous status: "
                f"{existing_document.ingestion_status}"
            )

            print(
                "Starting ingestion again..."
            )

            document = existing_document

        else:

            document = None


        # ==========================================
        # Load PDF
        # ==========================================

        print("Loading PDF...")

        pages = load_pdf(file_path)


        if not pages:

            raise ValueError(
                "No pages found in PDF."
            )


        # ==========================================
        # Clean Text
        # ==========================================

        print("Cleaning text...")

        cleaned_pages = []


        for page in pages:

            cleaned_text = clean_text(
                page["text"]
            )


            if cleaned_text:

                cleaned_pages.append({

                    "page_number":
                        page["page_number"],

                    "text":
                        cleaned_text

                })


        if not cleaned_pages:

            raise ValueError(
                "No readable text found in PDF."
            )


        # ==========================================
        # Create Chunks
        # ==========================================

        print("Creating chunks...")


        chunks = chunk_text(
            pages=cleaned_pages,
            source=source
        )


        total_chunks = len(chunks)


        print(
            f"Total chunks created: "
            f"{total_chunks}"
        )


        if not chunks:

            raise ValueError(
                "No chunks created from PDF."
            )


        # ==========================================
        # Create New Document
        # ==========================================

        if document is None:

            print(
                "Creating document..."
            )


            document = create_document(
                db=db,
                filename=source,
                file_hash=file_hash
            )


            print(
                f"Document created successfully. "
                f"Document ID: {document.id}"
            )


        # ==========================================
        # Calculate Total Batches
        # ==========================================

        total_batches = (

            total_chunks
            + INGESTION_BATCH_SIZE
            - 1

        ) // INGESTION_BATCH_SIZE


        # ==========================================
        # Process Chunks
        # ==========================================

        for start in range(

            0,
            total_chunks,
            INGESTION_BATCH_SIZE

        ):


            batch_number = (

                start
                // INGESTION_BATCH_SIZE

            ) + 1


            end = min(

                start
                + INGESTION_BATCH_SIZE,

                total_chunks

            )


            batch = chunks[start:end]


            print(
                f"\nProcessing batch "
                f"{batch_number}/"
                f"{total_batches}"
            )


            print(
                f"Chunks "
                f"{start + 1} to "
                f"{end}"
            )


            # ==========================================
            # Generate Embeddings
            # ==========================================

            print(
                "Generating embeddings..."
            )


            embedded_batch = embed_chunks(
                batch
            )


            # ==========================================
            # Save Chunks
            # ==========================================

            print(
                "Saving chunks to database..."
            )


            insert_chunks(

                db=db,

                document=document,

                chunks=embedded_batch

            )


            print(
                f"Batch "
                f"{batch_number} "
                f"completed."
            )


        # ==========================================
        # Mark As Completed
        # ==========================================

        mark_document_completed(
            db=db,
            document=document
        )


        # ==========================================
        # Success
        # ==========================================

        print(
            "\nIngestion completed successfully!"
        )

        print(
            f"Document ID: "
            f"{document.id}"
        )

        print(
            f"Total chunks saved: "
            f"{total_chunks}"
        )


        return document


    except Exception as error:


        # ==========================================
        # Mark As Failed
        # ==========================================

        if document is not None:

            try:

                mark_document_failed(
                    db=db,
                    document=document
                )

                print(
                    "Document marked as FAILED."
                )

            except Exception as status_error:

                print(
                    "Failed to update "
                    "document status:"
                )

                print(status_error)


        print(
            f"\nIngestion failed: "
            f"{str(error)}"
        )


        raise


    finally:

        db.close()


# ==========================================
# Run Script
# ==========================================

if __name__ == "__main__":

    pdf_path = (
        "data/raw/"
        "NIST_report.pdf"
    )

    ingest_pdf(pdf_path)