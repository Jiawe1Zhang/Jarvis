from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


class FaissVectorStore:
    """
    FAISS-backed vector store with optional persistence.
    """

    def __init__(
        self,
        index_factory: str = "Flat",
        persist_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        self.index_factory = index_factory
        self.persist_path = Path(persist_path) if persist_path else None
        self.metadata_path = (
            Path(metadata_path)
            if metadata_path
            else (self.persist_path.with_suffix(self.persist_path.suffix + ".meta") if self.persist_path else None)
        )
        self.index = None
        self.dim: Optional[int] = None
        self.next_id = 0
        self.id_to_doc: Dict[int, str] = {}
        # runtime meta for compatibility check
        self.embedding_model: Optional[str] = None
        self.chunk_strategy: Optional[str] = None
        self.data_signature: Optional[str] = None

    # --- Public API (matches in-memory store shape) ---
    def add_embedding(self, embedding: List[float], document: str) -> None:
        faiss, np = self._require_faiss()
        self._ensure_index(len(embedding), faiss)
        vector = np.asarray([embedding], dtype="float32")
        ids = self._next_ids(1, np)
        self.index.add_with_ids(vector, ids)
        self.id_to_doc[int(ids[0])] = document

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[str]:
        faiss, np = self._require_faiss()
        if not self.index or self.index.ntotal == 0:
            return []
        query = np.asarray([query_embedding], dtype="float32")
        distances, indices = self.index.search(query, top_k)
        results: List[str] = []
        for doc_id in indices[0]:
            if doc_id == -1:
                continue
            doc = self.id_to_doc.get(int(doc_id))
            if doc:
                results.append(doc)
        return results

    def size(self) -> int:
        return int(self.index.ntotal) if self.index else 0

    # --- Persistence ---
    def save(self) -> None:
        if not self.persist_path or not self.index:
            return
        faiss, _ = self._require_faiss()
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.persist_path))

        meta_path = self._meta_path()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "next_id": self.next_id,
            "dim": self.dim,
            "index_factory": self.index_factory,
            "id_to_doc": self.id_to_doc,
            "embedding_model": self.embedding_model,
            "chunk_strategy": self.chunk_strategy,
            "data_signature": self.data_signature,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        faiss, _ = self._require_faiss()
        if not self.persist_path or not self.persist_path.exists():
            return
        self.index = faiss.read_index(str(self.persist_path))
        meta_path = self._meta_path()
        if meta_path and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.next_id = int(meta.get("next_id", 0))
            self.dim = meta.get("dim")
            self.index_factory = meta.get("index_factory", self.index_factory)
            self.id_to_doc = {int(k): v for k, v in meta.get("id_to_doc", {}).items()}
            self.embedding_model = meta.get("embedding_model")
            self.chunk_strategy = meta.get("chunk_strategy")
            self.data_signature = meta.get("data_signature")
        else:
            self.next_id = int(self.index.ntotal)
            self.dim = self.index.d if self.index else None

    # --- Internal helpers ---
    def _ensure_index(self, dim: int, faiss) -> None:
        if self.index is not None:
            return
        self.dim = dim
        try:
            base_index = faiss.index_factory(dim, self.index_factory)
        except Exception:
            # Fallback: handle common strings like "FlatL2" or unknown -> use L2 flat
            if "ip" in self.index_factory.lower():
                base_index = faiss.IndexFlatIP(dim)
            else:
                base_index = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIDMap(base_index)

    def _next_ids(self, count: int, np):
        ids = np.arange(self.next_id, self.next_id + count, dtype="int64")
        self.next_id += count
        return ids

    def _meta_path(self) -> Path:
        if self.metadata_path:
            return self.metadata_path
        if self.persist_path:
            return Path(str(self.persist_path) + ".meta")
        raise RuntimeError("No metadata path configured")

    @staticmethod
    def _require_faiss():
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:  # broad to include missing np
            raise ImportError(
                "FAISS backend requested but `faiss` is not installed. "
                "On macOS (Apple Silicon) use: conda install -c conda-forge faiss-cpu. "
                "On Intel/mac: pip install faiss-cpu (if wheels available) or conda as above."
            ) from exc
        return faiss, np

    # --- Metadata helpers ---
    def set_meta_info(self, embedding_model: str, chunk_strategy: str, data_signature: str = "") -> None:
        self.embedding_model = embedding_model
        self.chunk_strategy = chunk_strategy
        self.data_signature = data_signature

    def is_compatible(self, embedding_model: str, chunk_strategy: str, data_signature: str = "") -> bool:
        if self.index is None:
            return False
        if self.embedding_model and self.embedding_model != embedding_model:
            return False
        if self.chunk_strategy and self.chunk_strategy != chunk_strategy:
            return False
        if self.data_signature and data_signature and self.data_signature != data_signature:
            return False
        return True

    def reset(self) -> None:
        """Clear index and metadata (used when meta mismatch)."""
        self.index = None
        self.dim = None
        self.next_id = 0
        self.id_to_doc = {}
