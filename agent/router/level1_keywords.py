"""
Level 1: Keyword / regex based router (strict).
Returns provisional decisions (may be partial):
- requires_rag: True/False/None
- tool_sets: List[str] or None (domains like CODING/PRODUCTIVITY/SEARCH)
"""
from typing import Dict, List, Optional, Set

# Signals for RAG intent (still permissive)
RAG_KEYWORDS: Set[str] = {
    "document",
    "doc",
    "pdf",
    "paper",
    "note",
    "notes",
    "summarize",
    "summary",
    "explain",
    "what is",
    "how does",
    "how to",
    "lookup",
    "find",
    "search",
    "reference",
}

# Signals for纯聊天
CHAT_KEYWORDS: Set[str] = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good evening",
    "joke",
    "poem",
    "story",
    "song",
    "who are you",
    "introduce yourself",
    "how are you",
}

# Strict tool phrases -> tool_sets (domains). Avoid single generic words.
TOOL_PHRASES: Dict[str, List[str]] = {
    # Coding
    "write python": ["CODING"],
    "python script": ["CODING"],
    "run python": ["CODING"],
    "python code": ["CODING"],
    "run code": ["CODING"],
    "execute code": ["CODING"],
    "shell command": ["CODING"],
    "run shell": ["CODING"],
    "edit file": ["CODING"],
    "modify file": ["CODING"],
    "file operation": ["CODING"],
    # Productivity
    "open notion": ["PRODUCTIVITY"],
    "update notion": ["PRODUCTIVITY"],
    "write to notion": ["PRODUCTIVITY"],
    "create notion": ["PRODUCTIVITY"],
    "schedule meeting": ["PRODUCTIVITY"],
    "calendar": ["PRODUCTIVITY"],
    "send email": ["PRODUCTIVITY"],
    # Search
    "search google": ["SEARCH"],
    "google search": ["SEARCH"],
    "look up": ["SEARCH"],
    "web search": ["SEARCH"],
    "weather forecast": ["SEARCH"],
    "stock price": ["SEARCH"],
}


def classify(query: str) -> Optional[Dict[str, object]]:
    """
    Return provisional routing decision; None if nothing detected.
    tool_sets is None when not determined; [] means explicitly none.
    """
    if not query:
        return None

    text = query.strip().lower()
    requires_rag: Optional[bool] = None
    tool_sets: Optional[List[str]] = None
    reasons: List[str] = []

    # RAG signals
    for kw in RAG_KEYWORDS:
        if kw in text:
            requires_rag = True
            reasons.append(f"Detected RAG keyword: {kw}")
            break

    # Tool signals (strict phrases)
    for phrase, domains in TOOL_PHRASES.items():
        if phrase in text:
            tool_sets = domains.copy()
            reasons.append(f"Detected tool phrase: {phrase}")
            break

    # Chat signals (only if nothing else triggered)
    if requires_rag is None and tool_sets is None and any(kw in text for kw in CHAT_KEYWORDS):
        return {
            "requires_rag": False,
            "tool_sets": [],
            "reasoning": "Detected casual chat intent.",
        }

    if requires_rag is None and tool_sets is None:
        return None

    return {
        "requires_rag": requires_rag,
        "tool_sets": tool_sets,
        "reasoning": "; ".join(reasons) if reasons else "Keyword routing hit.",
    }
