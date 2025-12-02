import json
from typing import List, Optional

from .chat_openai import ChatOpenAI
from .mcp_client import MCPClient
from .utils import log_title


class Agent:
    """
    Orchestrates the LLM conversation loop and MCP tool usage.
    """

    def __init__(
        self,
        model: str,
        mcp_clients: List[MCPClient],
        system_prompt: str = "",
        context: str = "",
    ) -> None:
        self.mcp_clients = mcp_clients
        self.system_prompt = system_prompt
        self.context = context
        self.model = model
        self.llm: Optional[ChatOpenAI] = None

    def init(self) -> None:
        log_title("TOOLS")
        for client in self.mcp_clients:
            client.init()
        tools = []
        for client in self.mcp_clients:
            tools.extend(client.get_tools())
        self.llm = ChatOpenAI(self.model, self.system_prompt, tools, self.context)

    def close(self) -> None:
        for client in self.mcp_clients:
            client.close()

    def invoke(self, prompt: str) -> str:
        if not self.llm:
            raise RuntimeError("Agent not initialized")

        response = self.llm.chat(prompt)
        while True:
            tool_calls = response.get("tool_calls", [])
            if tool_calls:
                for tool_call in tool_calls:
                    mcp = self._find_client(tool_call.name)
                    if not mcp:
                        self.llm.append_tool_result(tool_call.id, "Tool not found")
                        continue
                    log_title("TOOL USE")
                    print(f"Calling tool: {tool_call.name}")
                    print(f"Arguments: {tool_call.arguments}")
                    try:
                        result = mcp.call_tool(
                            tool_call.name,
                            json.loads(tool_call.arguments) if tool_call.arguments else {},
                        )
                    except Exception as exc:
                        result = {"error": str(exc)}
                    print(f"Result: {result}")
                    self.llm.append_tool_result(tool_call.id, json.dumps(result))
                response = self.llm.chat()
                continue
            self.close()
            return response.get("content", "")

    def _find_client(self, tool_name: str) -> Optional[MCPClient]:
        for client in self.mcp_clients:
            for tool in client.get_tools():
                if tool["name"] == tool_name:
                    return client
        return None
