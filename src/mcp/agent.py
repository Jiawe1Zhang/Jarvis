import json
from typing import List, Optional

from src.mcp.llm_client import ChatOpenAI
from src.mcp.mcp_client import MCPClient
from src.utils import log_title


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

    async def init(self) -> None:
        log_title("TOOLS")
        for client in self.mcp_clients:
            await client.init()

        # 拿到所有工具
        tools = []
        for client in self.mcp_clients:
            # the type of get_tools is a dict list
            client_tools = await client.get_tools()
            tools.extend(client_tools)

        # 初始化 llm
        self.llm = ChatOpenAI(self.model, self.system_prompt, tools, self.context)

    async def close(self) -> None:
        for client in self.mcp_clients:
            await client.close()

    async def invoke(self, prompt: str) -> str:
        if not self.llm:
            raise RuntimeError("Agent not initialized")

        # 拿到的返回response里有 content 和 tool_calls
        response = self.llm.chat(prompt)
        while True:
            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            # 纯原生 Function Calling，不使用兜底策略
            if tool_calls:
                print(f"[Native Function Calling] 检测到 {len(tool_calls)} 个工具调用")
                for tool_call in tool_calls:
                    tool_name = tool_call.name
                    tool_args = tool_call.arguments
                    tool_id = tool_call.id
                    try:
                        tool_args_dict = json.loads(tool_args) if tool_args else {}
                    except json.JSONDecodeError:
                        tool_args_dict = {}

                    mcp = await self._find_client(tool_name)
                    if not mcp:
                        self.llm.append_tool_result(tool_id, "Tool not found")
                        continue
                    log_title("TOOL USE")
                    print(f"Calling tool: {tool_name}")
                    print(f"Arguments: {tool_args_dict}")
                    try:
                        result = await mcp.call_tool(
                            tool_name,
                            tool_args_dict,
                        )
                        # Convert MCP content objects to serializable format
                        if isinstance(result, list):
                            serialized_result = []
                            for item in result:
                                if hasattr(item, 'text'):
                                    serialized_result.append(item.text)
                                elif hasattr(item, 'model_dump'):
                                    serialized_result.append(item.model_dump())
                                else:
                                    serialized_result.append(str(item))
                            result = serialized_result
                    except Exception as exc:
                        result = {"error": str(exc)}
                    print(f"Result: {result}")
                    self.llm.append_tool_result(tool_id, json.dumps(result))
                response = self.llm.chat()
                continue
            return content

    async def _find_client(self, tool_name: str) -> Optional[MCPClient]:
        for client in self.mcp_clients:
            tools = await client.get_tools()
            for tool in tools:
                if tool["name"] == tool_name:
                    return client
        return None
