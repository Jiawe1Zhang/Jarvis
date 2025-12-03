import os
import requests
from dataclasses import dataclass
from math import sqrt
from typing import List
from src.rag.vector_store import VectorStore

# 新增：导入 dotenv 来加载 .env 文件
from dotenv import load_dotenv

# 1. 加载环境变量
# 这行代码会自动寻找同目录下的 .env 文件并将配置注入到 os.environ 中
load_dotenv()

# --- Retriever (修改为读取环境变量) ---
class OllamaEmbeddingRetriever:
    def __init__(self) -> None:
        # 修改点 1: 从环境变量读取配置
        # 如果 .env 里没配，第二个参数是默认值，防止程序直接崩掉
        self.ollama_base_url = os.getenv("OLLAMA_EMBED_BASE_URL")
        self.embedding_model = os.getenv("OLLAMA_EMBED_MODEL")
        
        # 检查一下配置是否读取成功
        if not self.ollama_base_url:
            raise ValueError("❌ 未找到 OLLAMA_BASE_URL，请检查 .env 文件")
            
        print(f"⚙️  配置加载成功: URL={self.ollama_base_url}, Model={self.embedding_model}")
        
        self.vector_store = VectorStore()

    def embed_document(self, document: str) -> List[float]:
        print(f"🔄 正在向量化文档: {document[:10]}...")
        embedding = self._embed(document)
        if embedding: # 只有成功获取向量才存入
            self.vector_store.add_embedding(embedding, document)
        return embedding

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        print(f"🔎 正在处理提问: {query}")
        query_embedding = self._embed(query)
        if not query_embedding:
            return []
        return self.vector_store.search(query_embedding, top_k)

    def _embed(self, text: str) -> List[float]:
        # 修改点 2: 动态拼接 URL
        url = f"{self.ollama_base_url}/api/embeddings"
        
        try:
            response = requests.post(
                url, 
                json={
                    "model": self.embedding_model, # 使用环境变量里的模型名
                    "prompt": text
                }, 
                timeout=60
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"❌ 调用 Ollama 失败: {e}")
            print(f"   请检查 Ollama 是否在 {self.ollama_base_url} 运行")
            return []

if __name__ == "__main__":
    # 初始化时不再需要传参数，它会自动去读 .env
    app = OllamaEmbeddingRetriever()

    documents = [
        "环境变量 (.env) 是管理配置的安全方式。",
        "Ollama 允许在本地运行 bge-m3 等模型。",
        "Python 的 dotenv 库可以轻松读取 .env 文件。",
    ]

    print("\n--- 1. 存入文档 ---")
    for doc in documents:
        app.embed_document(doc)

    print("\n--- 2. 检索测试 ---")
    results = app.retrieve("怎么配置环境变量？")
    
    if results:
        print(f"✅ 结果: {results[0]}")
