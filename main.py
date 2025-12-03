import asyncio
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from src.mcp.agent import Agent
from src.mcp.mcp_client import MCPClient
from src.prompts.presets import SYSTEM_PROMPT
from src.config.loader import load_user_config
from src.utils import log_title
from src.rag.context import retrieve_context


def main() -> None:
    load_dotenv()  # 从 .env 加载 OPENAI_BASE_URL, OPENAI_API_KEY, OLLAMA_EMBED_BASE_URL 等
    cfg = load_user_config()

    # --- 读取配置 ---
    llm_cfg = cfg.get("llm", {})
    embed_cfg = cfg.get("embedding", {})
    knowledge_globs = cfg.get("knowledge_globs", ["knowledge/*.md"])
    task_template = cfg.get("task_template", "")

    # --- 输出目录 ---
    output_dir = Path.cwd() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_text = task_template.format(output_path=str(output_dir))

    # --- Embedding & RAG (base_url/api_key 从 .env 读取) ---
    context = retrieve_context(
        task=task_text,
        knowledge_globs=knowledge_globs,
        embed_model=embed_cfg.get("model", "bge-m3"),
    )

    # --- MCP Servers ---
    mcp_clients = []
    for server in cfg.get("mcp_servers", []):
        args = [arg.replace("{output_dir}", str(output_dir)) for arg in server.get("args", [])]
        mcp_clients.append(MCPClient(command=server["command"], args=args))

    # --- Agent (model 从配置读，其他从 .env) ---
    model_name = llm_cfg.get("model", "gpt-4o")
    agent = Agent(model_name, mcp_clients, context=context, system_prompt=SYSTEM_PROMPT)

    async def run_agent():
        await agent.init()
        try:
            await agent.invoke(task_text)
        finally:
            await agent.close()

    asyncio.run(run_agent())


if __name__ == "__main__":
    main()
