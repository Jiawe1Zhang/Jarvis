from dataclasses import dataclass
from math import sqrt
from typing import List


@dataclass
class VectorStoreItem:
    embedding: List[float]
    document: str


class VectorStore:
    """
    Extremely small in-memory vector store backed by cosine similarity.
    """

    def __init__(self) -> None:
        self._store: List[VectorStoreItem] = []

    def add_embedding(self, embedding: List[float], document: str) -> None:
        self._store.append(VectorStoreItem(embedding, document))

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[str]:
        scored = [
            (item.document, self._cosine_similarity(query_embedding, item.embedding))
            for item in self._store
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [document for document, _ in scored[:top_k]]

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sqrt(sum(a * a for a in vec_a))
        norm_b = sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
