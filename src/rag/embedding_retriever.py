import json
import os
from typing import List, Optional

import requests

from src.utils import log_title
from src.rag.vector_store import VectorStore


class EmbeddingRetriever:
    """
    Embedding 服务封装。
    
    支持两种模式：
    1. OpenAI 兼容 API（包括 OpenAI 官方、DeepBricks 等）
    2. Ollama 原生 API
    
    配置优先级：
    - 构造函数参数 > 环境变量
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.vector_store = VectorStore()
        
        # 确定使用哪种 API
        self._detect_api_type()

    def _detect_api_type(self) -> None:
        """检测应该使用哪种 API"""
        # 优先使用传入的参数
        if self.base_url:
            # 如果 URL 包含 /v1，认为是 OpenAI 兼容格式
            if "/v1" in self.base_url:
                self._api_type = "openai"
                self._endpoint = f"{self.base_url.rstrip('/')}/embeddings"
            else:
                # 否则认为是 Ollama 原生格式
                self._api_type = "ollama"
                self._endpoint = f"{self.base_url.rstrip('/')}/api/embeddings"
            return
        
        # 回退到环境变量
        openai_url = os.environ.get("EMBEDDING_BASE_URL")
        openai_key = os.environ.get("EMBEDDING_KEY")
        ollama_url = os.environ.get("OLLAMA_EMBED_BASE_URL")
        
        if openai_url and openai_key:
            self._api_type = "openai"
            self._endpoint = f"{openai_url.rstrip('/')}/embeddings"
            self.api_key = openai_key
        elif ollama_url:
            self._api_type = "ollama"
            self._endpoint = f"{ollama_url.rstrip('/')}/api/embeddings"
        else:
            raise RuntimeError(
                "Embedding 配置缺失。请设置以下任一组合：\n"
                "1. 构造函数传入 base_url\n"
                "2. 环境变量 EMBEDDING_BASE_URL + EMBEDDING_KEY\n"
                "3. 环境变量 OLLAMA_EMBED_BASE_URL"
            )

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
        if self._api_type == "openai":
            return self._embed_openai(text)
        else:
            return self._embed_ollama(text)

    def _embed_openai(self, text: str) -> List[float]:
        """OpenAI 兼容 API 格式"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        response = requests.post(
            self._endpoint,
            headers=headers,
            json={
                "model": self.model,
                "input": text,
                "encoding_format": "float",
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    def _embed_ollama(self, text: str) -> List[float]:
        """Ollama 原生 API 格式"""
        response = requests.post(
            self._endpoint,
            headers={"Content-Type": "application/json"},
            json={
                "model": self.model,
                "prompt": text,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]
