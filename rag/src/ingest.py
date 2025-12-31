"""Ingestion pipeline with sample penile SCC content for MVP."""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

from src.split import chunk_text


# Sample educational content about penile SCC for MVP
SAMPLE_CONTENT = {
    "penile-cancer-basics": """
    Penile cancer is a rare cancer that develops on the skin and tissues of the penis. 
    The most common type is squamous cell carcinoma, accounting for about 95% of cases.
    Risk factors include HPV infection, smoking, phimosis (tight foreskin), and poor hygiene.
    Early detection through self-examination and regular medical check-ups is important for better outcomes.
    """,
    
    "staging-system": """
    Penile cancer is staged using the TNM system:
    T-stage describes the size and extent of the primary tumor.
    N-stage refers to regional lymph node involvement (N0=none, N1=1-2 nodes, N2=3+ nodes, N3=fixed nodes).
    M-stage indicates distant metastasis.
    The lymph node status is the most important prognostic factor in penile cancer.
    """,
    
    "symptoms-diagnosis": """
    Common symptoms include a lump, ulcer, or bleeding on the penis, and discharge.
    Diagnosis typically involves physical examination and biopsy of suspicious lesions.
    Imaging like CT or MRI may be used to assess local spread and lymph node involvement.
    Early-stage disease often has excellent prognosis when treated promptly.
    """,
    
    "treatment-options": """
    Treatment depends on stage and grade. Options include:
    - Topical therapy (imiquimod, fluorouracil) for early superficial lesions
    - Circumcision and excisional biopsy for localized disease
    - Laser therapy or Mohs micrographic surgery for organ preservation
    - Partial or total penectomy for advanced tumors
    - Lymphadenectomy (removal of lymph nodes) if metastasis is suspected
    - Chemotherapy for advanced/metastatic disease
    Multidisciplinary team approach ensures optimal outcomes.
    """,
    
    "lymph-node-metastasis": """
    Lymph node involvement is the single most important prognostic factor in penile cancer.
    The inguinal lymph nodes are the first site of regional spread.
    Predictive features for lymph node metastasis include:
    - Higher tumor grade (G3-G4)
    - Deeper invasion (T2-T4)
    - Lymph-vascular invasion
    - Presence of carcinoma in situ
    Accurate staging of lymph nodes guides treatment decisions including surveillance vs. lymphadenectomy.
    """,
    
    "survival-prognosis": """
    5-year survival rates vary significantly by stage:
    - Stage I (T1-2 N0 M0): 80-90%
    - Stage II (T3 N0 M0): 50-70%
    - Stage III (Any T N1-2 M0): 20-40%
    - Stage IV (Any T N3 M0 or M1): <15%
    Early detection and complete lymph node staging improve outcomes.
    Histologic grade and depth of invasion are strong predictors of survival.
    """,
    
    "patient-support": """
    Living with or after penile cancer involves physical and psychological challenges.
    Support resources include:
    - Cancer support organizations and support groups
    - Mental health counseling for anxiety and depression
    - Sexual health counseling to address dysfunction concerns
    - Physical rehabilitation and wound care
    Regular follow-up with your oncology team is essential for monitoring.
    """,
    
    "questions-for-doctor": """
    Important questions to discuss with your healthcare team:
    - What is my cancer stage and what does it mean?
    - What are my treatment options and their side effects?
    - What is my survival prognosis?
    - Will I need lymph node surgery or imaging?
    - What follow-up care will I need?
    - Are there clinical trials I should consider?
    - How will treatment affect my sexual and urinary function?
    """
}


def ingest_sources():
    """Ingest sample content into ChromaDB."""
    # Initialize embedder
    print("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB
    chroma_dir = Path("data/chroma")
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.Client()
    
    collection = client.get_or_create_collection(
        name="penile_scc"
    )
    
    logs = []
    all_embeddings = []
    all_documents = []
    all_metadatas = []
    all_ids = []
    
    print(f"\nIngesting {len(SAMPLE_CONTENT)} sample documents...")
    
    global_chunk_id = 0
    for idx, (topic, content) in enumerate(SAMPLE_CONTENT.items(), 1):
        print(f"[{idx}/{len(SAMPLE_CONTENT)}] Processing {topic}...", end=" ")
        
        # Chunk
        chunks = chunk_text(
            content,
            chunk_size=512,
            overlap=100,
            source_url=f"Sample: {topic}",
            chunk_index_offset=0
        )
        
        if not chunks:
            print("SKIP")
            logs.append({"url": topic, "status": "SKIP", "reason": "no_chunks"})
            continue
        
        # Embed
        for chunk in chunks:
            text = chunk["text"]
            if len(text) < 20:
                continue
            
            embedding = embedder.encode(text).tolist()
            all_embeddings.append(embedding)
            all_documents.append(text)
            all_metadatas.append(chunk["metadata"])
            all_ids.append(f"chunk_{global_chunk_id}")
            global_chunk_id += 1
        
        print(f"OK ({len(chunks)} chunks)")
        logs.append({"url": topic, "status": "OK", "reason": None})
    
    # Batch insert
    if all_documents:
        print(f"\nInserting {len(all_documents)} chunks into ChromaDB...")
        collection.add(
            ids=all_ids,
            documents=all_documents,
            embeddings=all_embeddings,
            metadatas=all_metadatas
        )
        print("✓ Done!")
    
    # Write logs
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    log_file = reports_dir / "ingest_log.jsonl"
    
    with open(log_file, "w") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")
    
    print(f"\n✓ Ingest log saved to {log_file}")
    ok_count = sum(1 for l in logs if l['status'] == 'OK')
    print(f"✓ Successfully ingested {ok_count}/{len(SAMPLE_CONTENT)} documents")
    print("\n→ Ready! Run: streamlit run app.py")


if __name__ == "__main__":
    ingest_sources()
