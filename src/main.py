import os
from pathlib import Path

from dotenv import load_dotenv

from .agent import Agent
from .embedding_retriever import EmbeddingRetriever
from .mcp_client import MCPClient
from .utils import log_title


TASK = """
告诉我Antonette的信息,先从我给你的context中找到相关信息,总结后创作一个关于她的故事
把故事和她的基本信息保存到{output_path}/antonette.md,输出一个漂亮md文件
"""


def retrieve_context(task: str) -> str:
    knowledge_dir = Path.cwd() / "knowledge"
    retriever = EmbeddingRetriever("BAAI/bge-m3")
    for file_path in sorted(knowledge_dir.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")
        retriever.embed_document(content)
    context_blocks = retriever.retrieve(task, 3)
    context = "\n".join(context_blocks)
    log_title("CONTEXT")
    print(context)
    return context


def main() -> None:
    load_dotenv()
    output_dir = Path.cwd() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    task_text = TASK.format(output_path=str(output_dir))

    context = retrieve_context(task_text)

    fetch_mcp = MCPClient("mcp-server-fetch", "uvx", ["mcp-server-fetch"])
    file_mcp = MCPClient(
        "mcp-server-file",
        "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", str(output_dir)],
    )

    agent = Agent("openai/gpt-4o-mini", [fetch_mcp, file_mcp], context=context)
    agent.init()
    try:
        agent.invoke(task_text)
    finally:
        agent.close()


if __name__ == "__main__":
    main()
