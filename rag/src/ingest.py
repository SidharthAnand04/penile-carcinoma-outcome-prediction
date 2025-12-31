"""Ingestion pipeline for penile SCC using live URLs from data/sources.txt."""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

import requests
import trafilatura

from src.split import chunk_text
from src.db import get_client, ensure_collection_exists


def load_urls(path="data/sources.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def ingest_sources():
    """Ingests live web content from URLs in data/sources.txt into ChromaDB."""
    print("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    client = get_client()
    collection = ensure_collection_exists()
    logs = []
    all_embeddings = []
    all_documents = []
    all_metadatas = []
    all_ids = []
    urls = load_urls()
    print(f"\nIngesting {len(urls)} documents from URLs...")
    global_chunk_id = 0
    for idx, url in enumerate(urls, 1):
        print(f"[{idx}/{len(urls)}] Processing {url}...", end=" ")
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code != 200:
                print(f"SKIP (HTTP {resp.status_code})")
                logs.append({"url": url, "status": "SKIP", "reason": f"HTTP {resp.status_code}"})
                continue
            text = trafilatura.extract(resp.text, url=url)
            if not text or len(text) < 100:
                print("SKIP (no content)")
                logs.append({"url": url, "status": "SKIP", "reason": "no_content"})
                continue
            chunks = chunk_text(
                text,
                chunk_size=512,
                overlap=100,
                source_url=url,
                chunk_index_offset=0
            )
            if not chunks:
                print("SKIP (no chunks)")
                logs.append({"url": url, "status": "SKIP", "reason": "no_chunks"})
                continue
            for chunk in chunks:
                ctext = chunk["text"]
                if len(ctext) < 20:
                    continue
                embedding = embedder.encode(ctext).tolist()
                all_embeddings.append(embedding)
                all_documents.append(ctext)
                all_metadatas.append(chunk["metadata"])
                all_ids.append(f"chunk_{global_chunk_id}")
                global_chunk_id += 1
            print(f"OK ({len(chunks)} chunks)")
            logs.append({"url": url, "status": "OK", "reason": None})
        except Exception as e:
            print(f"SKIP (error: {e})")
            logs.append({"url": url, "status": "SKIP", "reason": str(e)})
    if all_documents:
        print(f"\nInserting {len(all_documents)} chunks into ChromaDB...")
        collection.add(
            ids=all_ids,
            documents=all_documents,
            embeddings=all_embeddings,
            metadatas=all_metadatas
        )
        print("Done!")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    log_file = reports_dir / "ingest_log.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")
    print(f"\nIngest log saved to {log_file}")
    ok_count = sum(1 for l in logs if l['status'] == 'OK')
    print(f"Successfully ingested {ok_count}/{len(urls)} documents")
    print("\nReady! Run: streamlit run app.py")

if __name__ == "__main__":
    ingest_sources()
