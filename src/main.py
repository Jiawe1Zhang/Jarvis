import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from src.agent import Agent
from src.embedding_retriever import EmbeddingRetriever
from src.MCP_Client import MCPClient
from src.utils import log_title


TASK = """
    总结并且创作一个关于Samantha的故事,你可以随意发挥想象力, 但要确保故事有趣且引人入胜.
    把故事和她的基本信息保存到{output_path}/samantha.md,输出一个漂亮md文件
"""

SYSTEM_PROMPT = """
You are an intelligent assistant equipped with specific tools.

### WORKFLOW
1. **Analyze Request**: Determine if the user's request requires performing actions (e.g., file I/O, web fetching) that your tools can handle.
2. **Select Tool**: If an action is needed, select the appropriate tool from your available list.
3. **Execute via Function Call**:
   - You MUST use the native **function calling** mechanism to invoke the tool.
   - Do NOT output the JSON arguments inside the conversation text (content).
   - The system will only execute tools passed through the dedicated `tool_calls` field, NOT those written in the message body.

### STRICT RULES
- **NO JSON IN TEXT**: Never output raw JSON or code blocks containing tool arguments in your response text.
- **Context Awareness**: Rely on the provided context. Do not invent file paths or data that do not exist.
- **Fabrication Prohibited**: Do not make up file paths or information not present in the context.
"""


def retrieve_context(task: str) -> str:
    knowledge_dir = Path.cwd() / "knowledge"
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "BAAI/bge-m3")
    retriever = EmbeddingRetriever(embed_model)
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
    # Ensure API key is set for local Ollama usage
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "ollama"

    output_dir = Path.cwd() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    task_text = TASK.format(output_path=str(output_dir))

    context = retrieve_context(task_text)

    fetch_mcp = MCPClient(command="uvx", args=["mcp-server-fetch"])
    file_mcp = MCPClient(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(output_dir)],
    )

    model_name = os.getenv("LLM_MODEL", "gpt-5")
    agent = Agent(model_name, [fetch_mcp, file_mcp], context=context, system_prompt=SYSTEM_PROMPT)

    async def run_agent():
        await agent.init()
        try:
            await agent.invoke(task_text)
        finally:
            await agent.close()

    asyncio.run(run_agent())


if __name__ == "__main__":
    main()
