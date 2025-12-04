from pathlib import Path
from typing import List

from rag.embedding_retriever import EmbeddingRetriever
from rag.loader import load_file
from utils import log_title


def retrieve_context(
    task: str,
    knowledge_globs: List[str],
    embed_model: str,
) -> str:
    """
    Embed knowledge sources and retrieve top matches for the given task.
    
    Args:
        task: 任务描述，用于检索相关上下文
        knowledge_globs: 知识文件的 glob 模式列表
        embed_model: embedding 模型名称
        
    Note:
        base_url 和 api_key 统一从 .env 环境变量读取
    """
    retriever = EmbeddingRetriever(model=embed_model)
    for pattern in knowledge_globs:
        for file_path in sorted(Path.cwd().glob(pattern)):
            if not file_path.is_file():
                continue
            
            # 使用统一的加载器处理不同格式的文件
            content = load_file(file_path)
            
            if content.strip():
                retriever.embed_document(content)
                
    context_blocks = retriever.retrieve(task, 3)
    context = "\n".join(context_blocks)
    log_title("CONTEXT")
    print(context)
    return context
