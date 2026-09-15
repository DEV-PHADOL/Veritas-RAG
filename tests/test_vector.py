import time

from sqlalchemy import text

from db.connection import SessionLocal
from ingestion.embedder import embed_query
from retrieval.vector import VectorRetriever


def measure_connection(db, connection_number):
    """
    Acquire a database connection and run SELECT 1.

    This lets us compare the first connection with
    a second connection in the same Python process.
    """

    print(
        f"\n========== CONNECTION {connection_number} =========="
    )

    # ------------------------------------------
    # Connection acquisition
    # ------------------------------------------

    print("Acquiring database connection...")

    connection_start = time.perf_counter()

    connection = db.connection()

    connection_time = (
        time.perf_counter() - connection_start
    )

    print(
        f"Connection acquisition time: "
        f"{connection_time:.4f}s"
    )

    # ------------------------------------------
    # SELECT 1
    # ------------------------------------------

    print("Running SELECT 1...")

    select_start = time.perf_counter()

    connection.execute(text("SELECT 1"))

    select_time = (
        time.perf_counter() - select_start
    )

    print(
        f"SELECT 1 time: {select_time:.4f}s"
    )

    return connection_time, select_time


def main():

    query = "What is cybersecurity awareness training?"

    # ==================================================
    # 1. SESSION CREATION
    # ==================================================

    print("\nCreating database session...")

    session_start = time.perf_counter()

    db = SessionLocal()

    session_time = (
        time.perf_counter() - session_start
    )

    print(
        f"Session creation time: "
        f"{session_time:.4f}s"
    )

    try:

        # ==================================================
        # 2. FIRST CONNECTION
        # ==================================================

        connection_1_time, select_1_time = (
            measure_connection(
                db=db,
                connection_number=1
            )
        )

        # ==================================================
        # 3. SECOND CONNECTION
        # ==================================================

        print(
            "\nClosing first connection..."
        )

        db.close()

        # Create a NEW session.
        #
        # The important point is that this is still
        # the SAME Python process, so SQLAlchemy's
        # connection pool can potentially reuse the
        # existing database connection.

        print(
            "\nCreating second database session..."
        )

        second_session_start = time.perf_counter()

        db = SessionLocal()

        second_session_time = (
            time.perf_counter()
            - second_session_start
        )

        print(
            f"Second session creation time: "
            f"{second_session_time:.4f}s"
        )

        connection_2_time, select_2_time = (
            measure_connection(
                db=db,
                connection_number=2
            )
        )

        # ==================================================
        # 4. QUERY EMBEDDING
        # ==================================================

        print("\nQuery:")
        print(query)

        print(
            "\nGenerating query embedding..."
        )

        embedding_start = time.perf_counter()

        query_embedding = embed_query(query)

        embedding_time = (
            time.perf_counter()
            - embedding_start
        )

        print(
            f"Query embedding time: "
            f"{embedding_time:.4f}s"
        )

        # ==================================================
        # 5. VECTOR SEARCH
        # ==================================================

        print(
            "\nSearching vector database..."
        )

        vector_start = time.perf_counter()

        retriever = VectorRetriever()

        results = retriever.search(
            db=db,
            query_embedding=query_embedding,
            top_k=5
        )

        vector_time = (
            time.perf_counter()
            - vector_start
        )

        print(
            f"Vector search time: "
            f"{vector_time:.4f}s"
        )

        # ==================================================
        # 6. VECTOR SEARCH RESULTS
        # ==================================================

        print("\nVector Search Results:")

        for result in results:

            print(
                f"\nChunk ID: {result['chunk_id']}"
            )

            print(
                f"Document ID: "
                f"{result['document_id']}"
            )

            print(
                f"Page Number: "
                f"{result['page_number']}"
            )

            print(
                f"Distance: "
                f"{result['score']}"
            )

            print(
                f"Content: "
                f"{result['content'][:500]}"
            )

            print("-" * 50)

        # ==================================================
        # 7. TIMING SUMMARY
        # ==================================================

        print("\n" + "=" * 60)
        print("DATABASE / VECTOR RETRIEVAL TIMING")
        print("=" * 60)

        print(
            f"First session creation    : "
            f"{session_time:.4f}s"
        )

        print(
            f"First connection          : "
            f"{connection_1_time:.4f}s"
        )

        print(
            f"First SELECT 1            : "
            f"{select_1_time:.4f}s"
        )

        print(
            f"Second session creation   : "
            f"{second_session_time:.4f}s"
        )

        print(
            f"Second connection         : "
            f"{connection_2_time:.4f}s"
        )

        print(
            f"Second SELECT 1           : "
            f"{select_2_time:.4f}s"
        )

        print(
            f"Query embedding           : "
            f"{embedding_time:.4f}s"
        )

        print(
            f"Vector search             : "
            f"{vector_time:.4f}s"
        )

        print("=" * 60)

    finally:

        db.close()


if __name__ == "__main__":
    main()