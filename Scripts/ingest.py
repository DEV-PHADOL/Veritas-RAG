
from pathlib import Path
import hashlib
import time

from ingestion.loader import load_pdf
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from ingestion.embedder import embed_chunks

from db.connection import SessionLocal

from db.repository import (
    create_document,
    insert_chunks,
    get_document_by_hash,
    delete_document_chunks,
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

    total_start = time.perf_counter()

    source = Path(file_path).name

    print("\n" + "=" * 60)
    print("STARTING DOCUMENT INGESTION")
    print("=" * 60)

    print(
        f"File: {source}"
    )

    # ==========================================
    # Calculate File Hash
    # ==========================================

    print("\nCalculating file hash...")

    hash_start = time.perf_counter()

    file_hash = calculate_file_hash(
        file_path
    )

    hash_time = (
        time.perf_counter()
        - hash_start
    )

    print(
        f"File hash time: "
        f"{hash_time:.4f}s"
    )

    # ==========================================
    # Database Session
    # ==========================================

    session_start = time.perf_counter()

    db = SessionLocal()

    session_time = (
        time.perf_counter()
        - session_start
    )

    print(
        f"DB session creation time: "
        f"{session_time:.4f}s"
    )

    document = None

    # These are initialized here so the
    # performance summary can always print.
    duplicate_check_time = 0.0
    load_time = 0.0
    clean_time = 0.0
    chunk_time = 0.0
    document_creation_time = 0.0
    total_embedding_time = 0.0
    total_insert_time = 0.0
    completed_status_time = 0.0

    try:

        # ==========================================
        # Check Existing Document
        # ==========================================

        print("\nChecking document...")

        duplicate_start = time.perf_counter()

        existing_document = get_document_by_hash(
            db=db,
            file_hash=file_hash
        )

        duplicate_check_time = (
            time.perf_counter()
            - duplicate_start
        )

        print(
            f"Duplicate check time: "
            f"{duplicate_check_time:.4f}s"
        )

        # ------------------------------------------
        # Already Successfully Ingested
        # ------------------------------------------

        if (
            existing_document
            and existing_document.ingestion_status
            == "COMPLETED"
        ):

            total_time = (
                time.perf_counter()
                - total_start
            )

            print(
                "\nDocument already fully ingested."
            )

            print(
                f"Document ID: "
                f"{existing_document.id}"
            )

            print(
                "Skipping ingestion."
            )

            print("\n" + "=" * 60)
            print("INGESTION SKIPPED")
            print("=" * 60)

            print(
                f"Total time: "
                f"{total_time:.4f}s"
            )

            print("=" * 60)

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
                "Cleaning previously inserted chunks..."
            )

            delete_document_chunks(
                db=db,
                document_id=existing_document.id
            )

            print(
                "Previous chunks deleted."
            )

            print(
                "Starting ingestion again..."
            )

            document = existing_document

        # ==========================================
        # Load PDF
        # ==========================================

        print("\nLoading PDF...")

        load_start = time.perf_counter()

        pages = load_pdf(
            file_path
        )

        load_time = (
            time.perf_counter()
            - load_start
        )

        print(
            f"PDF loading time: "
            f"{load_time:.4f}s"
        )

        print(
            f"Total pages: "
            f"{len(pages)}"
        )

        if not pages:

            raise ValueError(
                "No pages found in PDF."
            )

        # ==========================================
        # Clean Text
        # ==========================================

        print("\nCleaning text...")

        clean_start = time.perf_counter()

        cleaned_pages = []

        pages_with_text = 0
        empty_pages = 0

        for page in pages:

            cleaned_text = clean_text(
                page["text"]
            )

            if cleaned_text:

                pages_with_text += 1

                cleaned_pages.append({

                    "page_number":
                        page["page_number"],

                    "text":
                        cleaned_text
                })

            else:

                empty_pages += 1

        clean_time = (
            time.perf_counter()
            - clean_start
        )

        print(
            f"Text cleaning time: "
            f"{clean_time:.4f}s"
        )

        print(
            f"Pages with text: "
            f"{pages_with_text}"
        )

        print(
            f"Empty pages: "
            f"{empty_pages}"
        )

        if not cleaned_pages:

            raise ValueError(
                "No readable text found in PDF."
            )

        # ==========================================
        # Create Chunks
        # ==========================================

        print("\nCreating chunks...")

        chunk_start = time.perf_counter()

        chunks = chunk_text(
            pages=cleaned_pages,
            source=source
        )

        chunk_time = (
            time.perf_counter()
            - chunk_start
        )

        total_chunks = len(chunks)

        print(
            f"Chunking time: "
            f"{chunk_time:.4f}s"
        )

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

            print("\nCreating document...")

            document_start = time.perf_counter()

            document = create_document(
                db=db,
                filename=source,
                file_hash=file_hash
            )

            document_creation_time = (
                time.perf_counter()
                - document_start
            )

            print(
                f"Document creation time: "
                f"{document_creation_time:.4f}s"
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

        print(
            f"\nTotal batches: "
            f"{total_batches}"
        )

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

            print("\n" + "-" * 60)

            print(
                f"Processing batch "
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

            embedding_start = time.perf_counter()

            embedded_batch = embed_chunks(
                db=db,
                chunks=batch
            )

            embedding_time = (
                time.perf_counter()
                - embedding_start
            )

            total_embedding_time += (
                embedding_time
            )

            print(
                f"Embedding time: "
                f"{embedding_time:.4f}s"
            )

            # ==========================================
            # Save Chunks
            # ==========================================

            print(
                "Saving chunks to database..."
            )

            insert_start = time.perf_counter()

            insert_chunks(
                db=db,
                document=document,
                chunks=embedded_batch
            )

            insert_time = (
                time.perf_counter()
                - insert_start
            )

            total_insert_time += (
                insert_time
            )

            print(
                f"Database insert time: "
                f"{insert_time:.4f}s"
            )

            print(
                f"Batch "
                f"{batch_number} "
                f"completed."
            )

        # ==========================================
        # Mark As Completed
        # ==========================================

        print(
            "\nMarking document as COMPLETED..."
        )

        completed_start = time.perf_counter()

        mark_document_completed(
            db=db,
            document=document
        )

        completed_status_time = (
            time.perf_counter()
            - completed_start
        )

        print(
            f"Completion status update: "
            f"{completed_status_time:.4f}s"
        )

        # ==========================================
        # Total Time
        # ==========================================

        total_time = (
            time.perf_counter()
            - total_start
        )

        # ==========================================
        # Performance Summary
        # ==========================================

        print("\n" + "=" * 60)
        print("INGESTION PERFORMANCE SUMMARY")
        print("=" * 60)

        print(
            f"File hash              : "
            f"{hash_time:.4f}s"
        )

        print(
            f"DB session creation    : "
            f"{session_time:.4f}s"
        )

        print(
            f"Duplicate check        : "
            f"{duplicate_check_time:.4f}s"
        )

        print(
            f"PDF loading            : "
            f"{load_time:.4f}s"
        )

        print(
            f"Text cleaning          : "
            f"{clean_time:.4f}s"
        )

        print(
            f"Chunking               : "
            f"{chunk_time:.4f}s"
        )

        print(
            f"Document creation      : "
            f"{document_creation_time:.4f}s"
        )

        print(
            f"Total embedding        : "
            f"{total_embedding_time:.4f}s"
        )

        print(
            f"Total database inserts : "
            f"{total_insert_time:.4f}s"
        )

        print(
            f"Completion status      : "
            f"{completed_status_time:.4f}s"
        )

        print("-" * 60)

        print(
            f"TOTAL INGESTION        : "
            f"{total_time:.4f}s"
        )

        print("=" * 60)

        # ==========================================
        # Additional Statistics
        # ==========================================

        print("\nIngestion statistics:")

        print(
            f"Pages processed        : "
            f"{len(pages)}"
        )

        print(
            f"Pages with text        : "
            f"{pages_with_text}"
        )

        print(
            f"Empty pages            : "
            f"{empty_pages}"
        )

        print(
            f"Chunks created         : "
            f"{total_chunks}"
        )

        print(
            f"Batches processed      : "
            f"{total_batches}"
        )

        if total_batches > 0:

            print(
                f"Avg embedding/batch    : "
                f"{total_embedding_time / total_batches:.4f}s"
            )

            print(
                f"Avg DB insert/batch    : "
                f"{total_insert_time / total_batches:.4f}s"
            )

        if total_chunks > 0:

            print(
                f"Avg embedding/chunk    : "
                f"{total_embedding_time / total_chunks:.4f}s"
            )

            print(
                f"Avg DB insert/chunk    : "
                f"{total_insert_time / total_chunks:.4f}s"
            )

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

        total_time = (
            time.perf_counter()
            - total_start
        )

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

                print(
                    status_error
                )

        print(
            f"\nIngestion failed: "
            f"{str(error)}"
        )

        print(
            f"Time until failure: "
            f"{total_time:.4f}s"
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
