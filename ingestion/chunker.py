from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(pages: list[dict], source: str)->list[dict]:
    """
        Split cleaned PDF pages into smaller chunks while preserving
        page and source metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3200,
        chunk_overlap = 200,
    )
    
    chunks = []
    chunk_counter = 1
    
    for page in pages:
        page_number = page["page_number"]
        text = page["text"]
        
        if not text:
            continue
        
        page_chunks = splitter.split_text(text)
        
        for chunk in page_chunks:
            chunks.append({
                "chunk_id":f"chunk_{chunk_counter}",
                "text": chunk,
                "page_number": page_number,
                "source": source
            })
            
            chunk_counter += 1
            
    return chunks 