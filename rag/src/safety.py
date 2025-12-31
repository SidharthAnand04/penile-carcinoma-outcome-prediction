"""Safety guardrails for penile SCC learning tool."""

UNSAFE_PATTERNS = [
    "should i do",
    "what should i choose",
    "should i have surgery",
    "should i take",
    "dosage",
    "dose",
    "how much should i",
    "interpret my",
    "my biopsy",
    "my scan",
    "my results",
    "what does my",
    "is my tumor",
    "am i stage",
    "what's my prognosis",
]

def check_safety(query):
    """
    Check if query is safe to answer. Returns (is_safe, reason).
    
    Args:
        query: User's question string
    
    Returns:
        (bool, str): (is_safe, reason_if_unsafe)
    """
    query_lower = query.lower()
    
    for pattern in UNSAFE_PATTERNS:
        if pattern in query_lower:
            return False, f"Personal treatment advice not provided. Please consult your healthcare team."
    
    return True, ""


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
