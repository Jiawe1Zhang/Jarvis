"""
Level 2: Lightweight semantic router using anchor sentences and cosine similarity.
Proposes RAG intent and tool_sets separately (may be partial).
"""
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

# RAG intent anchors
RAG_ANCHORS: Sequence[str] = (
    "Summarize this document",
    "What does the paper say",
    "Explain these notes",
    "Retrieve facts from my knowledge base",
)

# Tool anchors (do not force RAG)
TOOL_ANCHORS: Sequence[Dict[str, object]] = (
    # Coding
    {"text": "Explain the code in detail", "tool_sets": ["CODING"]},
    {"text": "Write a python script", "tool_sets": ["CODING"]},
    {"text": "Run a shell command", "tool_sets": ["CODING"]},
    {"text": "Edit files on disk", "tool_sets": ["CODING"]},
    # Productivity
    {"text": "Update my notion database", "tool_sets": ["PRODUCTIVITY"]},
    {"text": "Schedule a meeting on calendar", "tool_sets": ["PRODUCTIVITY"]},
    {"text": "Send an email", "tool_sets": ["PRODUCTIVITY"]},
    # Search
    {"text": "Search the web for information", "tool_sets": ["SEARCH"]},
    {"text": "Check the weather forecast", "tool_sets": ["SEARCH"]},
    {"text": "Look up stock prices", "tool_sets": ["SEARCH"]},
)

SIMILARITY_THRESHOLD = 0.55


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _cosine_similarity(a: str, b: str) -> float:
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    ca = Counter(tokens_a)
    cb = Counter(tokens_b)
    dot = sum(ca[t] * cb[t] for t in set(ca) & set(cb))
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _dedupe(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def classify(query: str) -> Optional[Dict[str, object]]:
    if not query:
        return None

    requires_rag: Optional[bool] = None
    tool_sets: Optional[List[str]] = None
    reasons: List[str] = []

    # RAG anchors
    for text in RAG_ANCHORS:
        if _cosine_similarity(query, text) >= SIMILARITY_THRESHOLD:
            requires_rag = True
            reasons.append(f"Semantic RAG anchor: {text}")
            break

    # Tool anchors
    hits: List[str] = []
    for anchor in TOOL_ANCHORS:
        if _cosine_similarity(query, anchor["text"]) >= SIMILARITY_THRESHOLD:  # type: ignore[index]
            hits.extend(anchor.get("tool_sets", []))
            reasons.append(f"Semantic tool anchor: {anchor['text']}")

    if hits:
        tool_sets = _dedupe(hits)

    if requires_rag is None and tool_sets is None:
        return None

    return {
        "requires_rag": requires_rag,
        "tool_sets": tool_sets,
        "reasoning": "; ".join(reasons) if reasons else "Semantic routing hit.",
    }
