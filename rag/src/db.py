"""Shared ChromaDB client and collection storage."""

import chromadb

# Global client and collection (shared across app lifecycle)
_client = None
_collection = None


def get_client():
    """Get or create ChromaDB client."""
    global _client
    if _client is None:
        from chromadb.config import Settings
        # Use persistent storage for ChromaDB (ChromaDB 0.3.x)
        _client = chromadb.Client(Settings(persist_directory="rag/chroma_db"))
    return _client


def get_collection():
    """Get the penile_scc collection."""
    global _collection
    if _collection is None:
        client = get_client()
        try:
            _collection = client.get_collection(name="penile_scc")
        except:
            _collection = None
    return _collection


def ensure_collection_exists():
    """Create collection if it doesn't exist."""
    global _collection
    client = get_client()
    try:
        _collection = client.get_collection(name="penile_scc")
    except:
        _collection = client.create_collection(name="penile_scc")
    return _collection
