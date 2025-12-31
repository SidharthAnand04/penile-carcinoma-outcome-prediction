"""Chunking logic for document splitting with overlap and metadata preservation."""

def chunk_text(text, chunk_size=512, overlap=100, source_url=None, chunk_index_offset=0):
    """
    Split text into overlapping chunks.
    
    Args:
        text: Full document text
        chunk_size: Characters per chunk
        overlap: Characters of overlap between chunks
        source_url: Original source URL for metadata
        chunk_index_offset: Starting index for chunk numbering
    
    Returns:
        List of dicts: {"text": chunk_text, "metadata": {...}}
    """
    chunks = []
    stride = chunk_size - overlap
    
    i = 0
    chunk_num = chunk_index_offset
    while i < len(text):
        chunk_text = text[i:i+chunk_size]
        
        metadata = {
            "source": source_url or "unknown",
            "chunk_index": chunk_num,
            "start_char": i,
            "end_char": i + len(chunk_text)
        }
        
        chunks.append({
            "text": chunk_text.strip(),
            "metadata": metadata
        })
        
        i += stride
        chunk_num += 1
    
    return chunks
