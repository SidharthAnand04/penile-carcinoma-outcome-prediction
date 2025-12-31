"""Safety guardrails for penile SCC learning tool."""


# Lenient unsafe patterns: only block direct personal medical advice
UNSAFE_PATTERNS = [
    "should i", "what should i", "can i", "is it safe for me", "is it ok for me", "is it okay for me", "is it normal for me", "is it dangerous for me", "is it recommended for me"
]


# LLM-based safety check
import os
import requests
def llm_safety_check(query):
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
    if not api_key:
        return True, ""  # fallback: allow if no key
    system_prompt = (
        "You are a safety filter for a medical educational chatbot. "
        "If the following user question asks for personal medical advice, diagnosis, or anything unsafe, respond with 'UNSAFE'. "
        "If it is a general, safe, educational question, respond with 'SAFE'. "
        "Only respond with 'SAFE' or 'UNSAFE'."
    )
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 5,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip().upper()
        if "UNSAFE" in content:
            return False, "Personal treatment advice or unsafe request detected. Please consult your healthcare team."
        return True, ""
    except Exception:
        return True, ""  # fallback: allow if LLM check fails

def check_safety(query):
    """
    Check if query is safe to answer. Returns (is_safe, reason).
    
    Args:
        query: User's question string
    
    Returns:
        (bool, str): (is_safe, reason_if_unsafe)
    """
    query_lower = query.lower()
    # Block unsafe patterns (personalized/medical advice)
    for pattern in UNSAFE_PATTERNS:
        if pattern in query_lower:
            return False, "Personal treatment advice or interpretation is not provided. Please consult your healthcare team."
    # LLM-based safety check (no relevant keyword filter)
    return llm_safety_check(query)


def get_safe_redirect(query):
    """Suggest a safer version of the query."""
    suggestions = [
        "Try asking: 'What are the treatment options for penile cancer?'",
        "Try asking: 'What questions should I ask my doctor about penile cancer?'",
        "Try asking: 'What are the staging criteria for penile cancer?'",
        "Try asking: 'What are the symptoms of penile cancer?'",
    ]
    import random
    return random.choice(suggestions)
