import asyncio
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from agent.agent import Agent
from mcp_core.mcp_client import MCPClient
from config.loader import load_user_config
from utils import log_title
from utils.prompt_loader import load_prompt
from rag.context import retrieve_context
from utils.tracer import RunTracer
from utils.session_store import SessionStore
from datetime import datetime


def main() -> None:
    load_dotenv()  # Load OPENAI_BASE_URL, OPENAI_API_KEY, OLLAMA_EMBED_BASE_URL etc. from .env
    cfg = load_user_config("config/user_arxiv&notion.json")

    # --- Load Config ---
    llm_cfg = cfg["llm"]
    embed_cfg = cfg["embedding"]
    knowledge_globs = cfg["knowledge_globs"]
    task_template = cfg["task_template"]
    vector_store_cfg = cfg.get("vector_store", {})
    conversation_cfg = cfg.get("conversation_logging", {})
    rag_cfg = cfg.get("rag", {"enabled": True})

    # --- Output Directory ---
    output_dir = Path.cwd() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_text = task_template.format(output_path=str(output_dir))

    # --- Tracer Directory ---
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    tracer_dir = Path.cwd() / "logs" / run_id
    tracer = RunTracer(tracer_dir)
    tracer.info("run_start", {"task": task_text})

    # --- Session Store (optional) ---
    session_enabled = conversation_cfg.get("enabled", False)
    session_id = conversation_cfg.get("session_id") or run_id
    max_history_turns = conversation_cfg.get("max_history", 0)
    session_store = None
    if session_enabled:
        db_path = Path(conversation_cfg.get("db_path", "data/sessions.db"))
        session_store = SessionStore(db_path)

    # --- Embedding & RAG (base_url/api_key read from .env) ---
    context = ""
    if rag_cfg.get("enabled", True):
        # 这里拿到的context没有用户的query以及query 的改写, 就是纯rag的上下文
        context = retrieve_context(
            task=task_text,
            knowledge_globs=knowledge_globs,
            embed_model=embed_cfg["model"],
            chunking_strategy=embed_cfg["chunking_strategy"],
            enable_rewrite=embed_cfg["enable_query_rewrite"],
            rewrite_num_queries=embed_cfg.get("rewrite_num_queries", 3),
            llm_model=llm_cfg["model"],
            vector_store_config=vector_store_cfg,
            tracer=tracer,
        )
    else:
        tracer.info("rag_disabled", {"reason": "config.disabled"})

    # --- MCP Servers ---
    mcp_clients = []
    for server in cfg["mcp_servers"]:
        args = [arg.replace("{output_dir}", str(output_dir)) for arg in server["args"]]
        env = server.get("env")
        if env:
            resolved_env = {
                key: value.replace("{output_dir}", str(output_dir)) if isinstance(value, str) else value
                for key, value in env.items()
            }
        else:
            resolved_env = None

        mcp_clients.append(MCPClient(command=server["command"], args=args, env=resolved_env))

    # --- Agent (model read from config, others from .env) ---
    model_name = llm_cfg["model"]
    system_prompt = load_prompt("agent_system.md")
    if session_enabled:
        system_prompt += "\n\nYou are a stateful assistant with access to prior conversation history. Use it to stay consistent, avoid repetition, and reference past context when helpful."
    agent = Agent(
        model_name,
        mcp_clients,
        context=context,
        system_prompt=system_prompt,
        tracer=tracer,
        session_store=session_store,
        session_id=session_id,
        max_history_turns=max_history_turns,
    )

    async def run_agent():
        await agent.init()
        try:
            # 这里传入用户的任务
            await agent.invoke(task_text)
        finally:
            await agent.close()

    try:
        asyncio.run(run_agent())
    except Exception as exc:
        tracer.info("run_error", {"error": str(exc)})
        raise
    else:
        # flush history only on successful run
        agent.flush_history()


if __name__ == "__main__":
    main()
