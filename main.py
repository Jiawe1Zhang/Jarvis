import argparse
import asyncio
import json
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
from datetime import datetime, timezone
from utils.ui import get_ui
from agent.router import get_intent
from utils import log_title


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jarvis Agent Runner")
    parser.add_argument(
        "--config",
        default="config/user_arxiv&notion.json",
        help="Path to user config JSON (defaults to user_arxiv&notion.json)",
    )
    parser.add_argument(
        "--mcp-registry",
        default="config/mcp_servers.json",
        help="Path to MCP server registry (domain-aware); defaults to config/mcp_servers.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()  # Load OPENAI_BASE_URL, OPENAI_API_KEY, OLLAMA_EMBED_BASE_URL etc. from .env
    cfg = load_user_config(args.config)
    mcp_registry_path = Path(args.mcp_registry)
    if not mcp_registry_path.exists():
        raise FileNotFoundError(f"MCP registry not found: {mcp_registry_path}")
    with mcp_registry_path.open("r", encoding="utf-8") as f:
        mcp_registry = json.load(f)

    # --- Load Config ---
    llm_cfg = cfg["llm"]
    embed_cfg = cfg["embedding"]
    knowledge_globs = cfg["knowledge_globs"]
    query_template = cfg.get("query_template") or cfg["task_template"]
    vector_store_cfg = cfg.get("vector_store", {})
    conversation_cfg = cfg.get("conversation_logging", {})
    # knowledge toggle (new key "knowledge"; fallback to legacy "rag")
    knowledge_cfg = cfg.get("knowledge") or cfg.get("rag", {"enabled": True})
    ui_cfg = cfg.get("tui", {"enabled": False})
    ui = get_ui(ui_cfg.get("enabled", False))

    # --- Output Directory ---
    output_dir = Path.cwd() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    # --- Query (prompt from terminal; fallback to config template) ---
    try:
        raw_query = input("请输入查询/问题（留空则使用配置模板）：").strip()
    except EOFError:
        raw_query = ""
    if not raw_query:
        raw_query = query_template
    try:
        query_text = raw_query.format(output_path=str(output_dir))
    except Exception:
        query_text = raw_query

    # --- Tracer Directory ---
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tracer_dir = Path.cwd() / "logs" / run_id
    tracer = RunTracer(tracer_dir)
    tracer.info("run_start", {"query": query_text})

    # --- Intent Router (optional) ---
    intent_router_cfg = cfg.get("intent_router", {"enabled": False})
    intent_router_enabled = intent_router_cfg.get("enabled", False)
    intent_result = {
        "requires_rag": True,
        "tool_sets": [],
        "specific_tools": [],
        "reasoning": "Router disabled; default to RAG.",
    }

    # --- Decide whether to use external knowledge ---
    knowledge_enabled = knowledge_cfg.get("enabled", True)
    use_rag = knowledge_enabled and intent_result["requires_rag"]

    if intent_router_enabled:
        try:
            intent_result = get_intent(query_text, available_servers=mcp_registry)
            tracer.info("intent_router", {"intent": intent_result})
        except Exception as exc:
            tracer.info("intent_router_error", {"error": str(exc)})
            intent_result = {
                "requires_rag": True,
                "tool_sets": [],
                "specific_tools": [],
                "reasoning": "Router error; default to RAG.",
            }
        use_rag = knowledge_enabled and bool(intent_result.get("requires_rag"))
    else:
        # no intent router: fall back to config toggle only
        use_rag = knowledge_enabled

    if not knowledge_enabled:
        tracer.info("knowledge_disabled", {"reason": "config.disabled"})
    elif knowledge_enabled and not use_rag and intent_router_enabled:
        tracer.info("rag_disabled", {"reason": "intent_router_requires_rag_false"})

    # Developer-facing logging when TUI is off
    if not ui.enabled:
        log_title("INTENT ROUTER")
        print(f"router_enabled: {intent_router_enabled}")
        print(f"intent: {intent_result}")
        print(f"use_external_knowledge: {use_rag}")

    # --- Session Store (optional) ---
    session_enabled = conversation_cfg.get("enabled", False)
    session_id = conversation_cfg.get("session_id") or run_id
    max_history_turns = conversation_cfg.get("max_history", 0)
    session_store = None
    if session_enabled:
        db_path = Path(conversation_cfg.get("db_path", "data/sessions.db"))
        session_store = SessionStore(db_path)
        # Preload turn count for visibility
        existing_turns = session_store.load_turns(session_id, limit=max_history_turns or 0)
        current_turns = len(existing_turns)
        max_turns_display = max_history_turns if max_history_turns else "unlimited"
        if ui.enabled:
            ui.log("System", f"Session {session_id}: {current_turns} stored / max {max_turns_display}")
        else:
            log_title("SESSION")
            print(f"Session {session_id}: {current_turns} stored / max {max_turns_display}")

    # --- Embedding & RAG (base_url/api_key read from .env) ---
    context = ""
    if use_rag:
        # 这里拿到的context没有用户的query以及query 的改写, 就是纯rag的上下文
        context = retrieve_context(
            task=query_text,
            knowledge_globs=knowledge_globs,
            embed_model=embed_cfg["model"],
            chunking_strategy=embed_cfg["chunking_strategy"],
            enable_rewrite=embed_cfg["enable_query_rewrite"],
            rewrite_num_queries=embed_cfg.get("rewrite_num_queries", 3),
            llm_model=llm_cfg["model"],
            vector_store_config=vector_store_cfg,
            tracer=tracer,
            ui=ui,
        )
    else:
        tracer.info(
            "rag_disabled",
            {"reason": "config.disabled" if not knowledge_enabled else "intent_router_requires_rag_false"},
        )

    # --- MCP Servers (domain-aware filtering) ---
    def _select_servers(intent: dict, registry: list[dict]) -> list[dict]:
        if not intent_router_enabled:
            return registry
        domains = set(intent.get("tool_sets") or intent.get("tool_domains") or [])
        specific = set(intent.get("specific_tools") or [])
        selected = []
        for server in registry:
            srv_domains = set(server.get("domains") or [])
            srv_tools = set(server.get("tools") or [])
            if specific and srv_tools and specific.intersection(srv_tools):
                selected.append(server)
                continue
            if domains and srv_domains.intersection(domains):
                selected.append(server)
        # If intent provided no tool hints, load none (router decided tools not needed)
        if not domains and not specific:
            return []
        return selected

    selected_servers = _select_servers(intent_result, mcp_registry)

    if not ui.enabled:
        log_title("MCP SERVER SELECTION")
        print(f"available: {[s.get('name') for s in mcp_registry]}")
        print(f"selected: {[s.get('name') for s in selected_servers]}")

    mcp_clients = []
    for server in selected_servers:
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
        ui=ui,
    )

    async def run_agent():
        await agent.init()
        try:
            # 这里传入用户的任务
            await agent.invoke(query_text)
        finally:
            await agent.close()

    try:
        with ui.live():
            if ui.enabled:
                ui.log("User", query_text)
            asyncio.run(run_agent())
    except Exception as exc:
        tracer.info("run_error", {"error": str(exc)})
        raise
    else:
        # flush history only on successful run
        agent.flush_history()


if __name__ == "__main__":
    main()
