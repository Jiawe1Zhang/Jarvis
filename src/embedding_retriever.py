import json
import os
from typing import List

import requests

from .utils import log_title
from .vector_store import VectorStore


class EmbeddingRetriever:
    """
    Lightweight wrapper around an external embedding service.
    """

    def __init__(self, embedding_model: str) -> None:
        self.embedding_model = embedding_model
        self.vector_store = VectorStore()

    def embed_document(self, document: str) -> List[float]:
        log_title("EMBEDDING DOCUMENT")
        embedding = self._embed(document)
        self.vector_store.add_embedding(embedding, document)
        return embedding

    def embed_query(self, query: str) -> List[float]:
        log_title("EMBEDDING QUERY")
        return self._embed(query)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self.embed_query(query)
        return self.vector_store.search(query_embedding, top_k)

    def _embed(self, text: str) -> List[float]:
        # 1. Try OpenAI-compatible config first
        url = os.environ.get("EMBEDDING_BASE_URL")
        api_key = os.environ.get("EMBEDDING_KEY")

        if url and api_key:
            response = requests.post(
                f"{url.rstrip('/')}/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                data=json.dumps(
                    {
                        "model": self.embedding_model,
                        "input": text,
                        "encoding_format": "float",
                    }
                ),
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]

        # 2. Fallback to Ollama config
        ollama_url = os.environ.get("OLLAMA_EMBED_BASE_URL")
        if ollama_url:
            # Ollama native API expects "prompt" instead of "input"
            response = requests.post(
                f"{ollama_url.rstrip('/')}/api/embeddings",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "model": self.embedding_model,
                    "prompt": text,
                }),
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]

        raise RuntimeError("EMBEDDING_BASE_URL/KEY or OLLAMA_EMBED_BASE_URL must be set")
