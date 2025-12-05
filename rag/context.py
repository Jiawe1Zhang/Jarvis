from pathlib import Path
from typing import List, Optional

from agent.llm_client import SimpleLLMClient
from rag.embedding_retriever import EmbeddingRetriever
from rag.loader import load_file
from rag.query_rewriter import QueryRewriter
from utils import log_title


def retrieve_context(
    task: str,
    knowledge_globs: List[str],
    embed_model: str,
    chunking_strategy: str = "whole",
    enable_rewrite: bool = False,
    rewrite_num_queries: int = 3,
    llm_model: Optional[str] = None,
    vector_store_config: Optional[dict] = None,
    tracer=None,
) -> str:
    """
    Embed knowledge sources and retrieve top matches for the given task.
    
    Args:
        task: Task description, used to retrieve relevant context
        knowledge_globs: List of glob patterns for knowledge files
        embed_model: Embedding model name
        chunking_strategy: Chunking strategy ("whole" or "recursive")
        enable_rewrite: Whether to enable query rewriting
        rewrite_num_queries: Number of queries to generate if rewriting is enabled
        llm_model: LLM model name for query rewriting
        vector_store_config: Vector store backend config (memory/faiss, paths, etc.)
        
    Note:
        base_url and api_key are read from .env environment variables
    """
    data_signature = _compute_data_signature(knowledge_globs)
    retriever = EmbeddingRetriever(
        model=embed_model,
        chunking_strategy=chunking_strategy,
        vector_store_config=vector_store_config,
    )
    retriever.set_meta_info(embed_model, chunking_strategy, data_signature)
    retriever.ensure_compatibility(embed_model, chunking_strategy, data_signature)
    if tracer:
        tracer.log_event(
            {
                "type": "context_start",
                "data_signature": data_signature,
                "reuse_index": False,
            }
        )
    reuse_index = retriever.has_ready_index(embed_model, chunking_strategy, data_signature)
    if tracer and reuse_index:
        tracer.log_event({"type": "context_reuse_index", "size": getattr(retriever.vector_store, "size", lambda: None)() if retriever.vector_store else None})
    if not reuse_index:
        for pattern in knowledge_globs:
            for file_path in sorted(Path.cwd().glob(pattern)):
                if not file_path.is_file():
                    continue
                
                # Use unified loader to handle files of different formats
                content = load_file(file_path)
                
                if content.strip():
                    # doc_id is file name
                    retriever.embed_document(content)
            
    # --- Retrieval Logic ---
    search_queries = [task]
    if enable_rewrite and llm_model:
        print("Rewriting query...")
        rewriter = QueryRewriter(llm_model)
        rewritten_queries = rewriter.rewrite(task, num_queries=rewrite_num_queries)
        print(f"Rewritten queries: {rewritten_queries}")
        search_queries.extend(rewritten_queries)

    all_results = []
    seen_texts = set()
    
    for query in search_queries:
        # If rewrite is enabled, query fewer items per query to avoid too long context
        k = 2 if enable_rewrite else 3
        results = retriever.retrieve(query, top_k=k)
        
        for text in results:
            if text not in seen_texts:
                all_results.append(text)
                seen_texts.add(text)

    context = "\n\n".join(all_results)
    log_title("CONTEXT")
    print(context)
    # 保存向量索引（仅对支持持久化的后端有效，例如 FAISS）
    retriever.save_if_possible()
    if tracer:
        tracer.log_event({"type": "context_done", "chunks": len(all_results)})
    return context


def _compute_data_signature(knowledge_globs: List[str]) -> str:
    """
    Lightweight signature based on file path + mtime to detect knowledge changes.
    """
    parts = []
    for pattern in knowledge_globs:
        for file_path in sorted(Path.cwd().glob(pattern)):
            if not file_path.is_file():
                continue
            try:
                mtime = file_path.stat().st_mtime
                parts.append(f"{file_path.relative_to(Path.cwd())}:{mtime}")
            except Exception:
                continue
    return "|".join(parts)
