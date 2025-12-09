import json
from typing import List, Optional

from agent.llm_client import ChatOpenAI
from mcp_core.mcp_client import MCPClient
from utils import log_title
from utils.ui import BaseUI


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
        tracer=None,
        session_store=None,
        session_id: str = "default",
        max_history_turns: Optional[int] = None,
        ui: Optional[BaseUI] = None,
    ) -> None:
        self.mcp_clients = mcp_clients
        self.system_prompt = system_prompt
        self.context = context
        self.model = model
        self.llm: Optional[ChatOpenAI] = None
        self.tracer = tracer
        self.session_store = session_store
        self.session_id = session_id
        self.max_history_turns = max_history_turns
        self.ui = ui or BaseUI()

    async def init(self) -> None:
        if self.ui.enabled:
            self.ui.stage("Initialization", "in_progress")
        else:
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
        self.llm = ChatOpenAI(
            self.model,
            self.system_prompt,
            tools,
            self.context,
            tracer=self.tracer,
            session_store=self.session_store,
            session_id=self.session_id,
            max_history_turns=self.max_history_turns,
            ui=self.ui,
        )
        if self.ui.enabled:
            self.ui.stage("Initialization", "completed")

    async def close(self) -> None:
        for client in self.mcp_clients:
            await client.close()

    async def invoke(self, prompt: str) -> str:
        if not self.llm:
            raise RuntimeError("Agent not initialized")

        if self.ui.enabled:
            self.ui.stage("Agent Reasoning", "in_progress")
            self.ui.log("User", prompt)

        # 拿到的返回response里有 content 和 tool_calls
        response = self.llm.chat(prompt)
        while True:
            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            # 如果有内容，先打印出来；若为空但有工具调用，打印占位
            if content:
                if self.ui.enabled:
                    self.ui.log("Model", content)
                else:
                    log_title("LLM OUTPUT")
                    print(f"[MODEL] {content}")
            elif tool_calls:
                if not self.ui.enabled:
                    print("[MODEL] (tool call, no content)")
            # 纯原生 Function Calling，不使用兜底策略
            if tool_calls:
                if self.ui.enabled:
                    self.ui.stage("Agent Reasoning", "completed")
                    self.ui.stage("Tool Execution", "in_progress")
                else:
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
                    if self.ui.enabled:
                        self.ui.tool(tool_name, tool_args_dict)
                    else:
                        log_title("TOOL USE")
                        print(f"Calling tool: {tool_name}")
                        print(f"Arguments: {self._preview(tool_args_dict)}")
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
                    if self.ui.enabled:
                        preview = self._preview(result)
                        self.ui.detail(f"Result: {tool_name}", preview)
                        self.ui.tool(tool_name, tool_args_dict, preview)
                    else:
                        print(f"Result (preview): {self._preview(result)}")
                    if self.tracer:
                        self.tracer.log_event(
                            {
                                "type": "tool_call",
                                "tool": tool_name,
                                "args": tool_args_dict,
                                "result": result,
                            }
                        )
                    self.llm.append_tool_result(tool_id, json.dumps(result))
                if self.ui.enabled:
                    self.ui.stage("Tool Execution", "completed")
                    self.ui.stage("Agent Reasoning", "in_progress")
                response = self.llm.chat()
                continue
            if self.ui.enabled:
                self.ui.stage("Agent Reasoning", "completed")
                self.ui.stage("Final Response", "completed")
            return content

    def flush_history(self) -> None:
        if self.llm and hasattr(self.llm, "flush_history"):
            self.llm.flush_history()

    async def _find_client(self, tool_name: str) -> Optional[MCPClient]:
        for client in self.mcp_clients:
            tools = await client.get_tools()
            for tool in tools:
                if tool["name"] == tool_name:
                    return client
        return None

    @staticmethod
    def _preview(data: object, limit: int = 400) -> str:
        """
        Compact preview for logging to avoid flooding the terminal.
        """
        try:
            text = json.dumps(data, ensure_ascii=False)
        except Exception:
            text = str(data)
        if len(text) > limit:
            return text[:limit] + "...(truncated)"
        return text
