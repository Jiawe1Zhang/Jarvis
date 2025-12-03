from pathlib import Path
from typing import List

from src.rag.embedding_retriever import EmbeddingRetriever
from src.utils import log_title


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
            content = file_path.read_text(encoding="utf-8")
            retriever.embed_document(content)
    context_blocks = retriever.retrieve(task, 3)
    context = "\n".join(context_blocks)
    log_title("CONTEXT")
    print(context)
    return context
