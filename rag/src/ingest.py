"""Ingestion pipeline: fetch URLs, chunk, embed, and store in ChromaDB."""

import os
import json
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from src.split import chunk_text


def fetch_url_content(url, timeout=15):
    """Fetch and extract text from a URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Try trafilatura first
        try:
            import trafilatura
            content = trafilatura.extract(response.text)
            if content:
                return content
        except (ImportError, Exception):
            pass
        
        # Fallback: simple text extraction
        import re
        text = response.text
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text if text else None
    
    except Exception as e:
        return None


def ingest_sources():
    """Main ingestion pipeline."""
    sources_file = Path("data/sources.txt")
    if not sources_file.exists():
        print(f"ERROR: {sources_file} not found")
        return
    
    urls = [line.strip() for line in sources_file.read_text().split('\n') if line.strip()]
    
    # Initialize embedder
    print("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB
    chroma_dir = Path("data/chroma")
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb_parquet",
        persist_directory=str(chroma_dir),
        anonymized_telemetry=False,
    ))
    
    collection = client.get_or_create_collection(
        name="penile_scc",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Ingestion log
    logs = []
    all_embeddings = []
    all_documents = []
    all_metadatas = []
    all_ids = []
    
    print(f"\nIngesting {len(urls)} URLs...")
    
    global_chunk_id = 0
    for idx, url in enumerate(urls, 1):
        print(f"[{idx}/{len(urls)}] Fetching {url[:60]}...", end=" ")
        
        content = fetch_url_content(url)
        if not content:
            print("SKIP (fetch failed)")
            logs.append({"url": url, "status": "SKIP", "reason": "fetch_failed"})
            continue
        
        # Chunk
        chunks = chunk_text(
            content,
            chunk_size=512,
            overlap=100,
            source_url=url,
            chunk_index_offset=0
        )
        
        if not chunks:
            print("SKIP (no chunks)")
            logs.append({"url": url, "status": "SKIP", "reason": "no_chunks"})
            continue
        
        # Embed and prepare for batch insert
        for chunk in chunks:
            text = chunk["text"]
            if len(text) < 20:  # Skip tiny chunks
                continue
            
            embedding = embedder.encode(text).tolist()
            all_embeddings.append(embedding)
            all_documents.append(text)
            all_metadatas.append(chunk["metadata"])
            all_ids.append(f"chunk_{global_chunk_id}")
            global_chunk_id += 1
        
        print(f"OK ({len(chunks)} chunks)")
        logs.append({"url": url, "status": "OK", "reason": None})
    
    # Batch insert to ChromaDB
    if all_documents:
        print(f"\nInserting {len(all_documents)} chunks into ChromaDB...")
        collection.upsert(
            ids=all_ids,
            documents=all_documents,
            embeddings=all_embeddings,
            metadatas=all_metadatas
        )
        client.persist()
        print("Done!")
    
    # Write logs
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    log_file = reports_dir / "ingest_log.jsonl"
    
    with open(log_file, "w") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")
    
    print(f"\nIngest log saved to {log_file}")
    print(f"Successfully ingested {sum(1 for l in logs if l['status'] == 'OK')}/{len(urls)} URLs")


if __name__ == "__main__":
    ingest_sources()
