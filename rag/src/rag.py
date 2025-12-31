"""RAG pipeline: retrieve chunks and call OpenRouter LLM."""

import os
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
import requests
import json
import dotenv

from src.db import get_collection


def load_dotenv():
    dotenv_path = Path(__file__).parent.parent / ".env"
    dotenv.load_dotenv(dotenv_path)

load_dotenv()


class PenileSCCRAG:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get shared collection
        self.collection = get_collection()
        if self.collection is None:
            raise RuntimeError(
                "ChromaDB collection not found. Run: python -m src.ingest"
            )
        
        # OpenRouter settings (use st.secrets for Streamlit Cloud)
        self.api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", "")).strip()
        self.model = st.secrets.get("OPENROUTER_MODEL", os.getenv("OPENROUTER_MODEL", "openrouter/auto"))
        self.site_url = st.secrets.get("OPENROUTER_SITE_URL", os.getenv("OPENROUTER_SITE_URL", "")).strip()
        self.app_name = st.secrets.get("OPENROUTER_APP_NAME", os.getenv("OPENROUTER_APP_NAME", "")).strip()
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set in secrets or .env")

    
    def retrieve(self, query, k=7):
        """Retrieve top-k relevant chunks."""
        embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
        )
        
        chunks = []
        if results and results["documents"]:
            for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                chunks.append({
                    "text": doc,
                    "metadata": meta,
                    "rank": i + 1
                })
        
        return chunks
    
    def call_openrouter(self, system_prompt, user_message, temperature=0.7, max_tokens=1024, request_id=None):
        """Call OpenRouter API with proper headers and request tracking."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Add a custom header to identify the request source/location
            "X-Request-Source": f"rag.py:call_openrouter{f'|id={request_id}' if request_id else ''}"
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        }
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {e}")
    
    def answer_question(self, query, audience="family", depth="quick", request_id=None):
        """Generate RAG answer with citations."""
        # Retrieve
        chunks = self.retrieve(query, k=7)
        
        if not chunks:
            return {
                "answer": "I don't have enough information to answer this question from the sources. Please try a different query or ask your healthcare provider.",
                "chunks": [],
                "citations": []
            }
        
        # Build context with numbered citations
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"[{chunk['rank']}] {chunk['text']}")
        context = "\n\n".join(context_parts)
        
        # System prompt
        audience_note = "Keep language simple and avoid medical jargon." if audience == "family" else "Use clinical language."
        depth_note = "Keep answer concise (2-3 sentences)." if depth == "quick" else "Provide thorough explanation with context."
        
        system_prompt = f"""You are an educational assistant about penile cancer. Answer based ONLY on the provided sources.

    Audience: {audience} ({audience_note})
    Depth: {depth} ({depth_note})

    Rules:
    - Every statement must cite sources using [1], [2], etc. matching the context numbers
    - The numbered sources and URLs you use MUST be real and come ONLY from the provided context below. Do NOT make up or hallucinate any sources or links.
    - Do NOT output a "Sources:" section. The user interface will show the real sources separately.
    - If information is insufficient, say so clearly
    - End with 3-7 "Questions to ask your doctor" bullets
    - NEVER give personal medical advice
    - Be educational and grounded

    Context (numbered sources, each with a real URL):
    {context}"""
        
        user_message = f"Question: {query}"
        
        # Call LLM
        answer_text = self.call_openrouter(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.5,
            max_tokens=2048,
            request_id=request_id
        )
        
        return {
            "answer": answer_text,
            "chunks": chunks,
            "citations": [c["metadata"]["source"] for c in chunks]
        }
