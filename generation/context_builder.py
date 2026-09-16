def build_context(
    results: list[dict]
) -> str:

    if not results:
        return ""

    context_parts = []

    for index, result in enumerate(results, start=1):

        content = result["content"]
        page_number = result["page_number"]
        document_id = result["document_id"]
        chunk_id = result["chunk_id"]

        context_part = (
            f"[Source {index}]\n"
            f"Document ID: {document_id}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Page Number: {page_number}\n"
            f"Content:\n{content}\n"
        )

        context_parts.append(context_part)

    return "\n\n".join(context_parts)