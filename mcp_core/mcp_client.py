from typing import Optional, List, Any, Dict
from contextlib import AsyncExitStack
import json
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv

load_dotenv()


# MCP client build refer to https://modelcontextprotocol.io/docs/develop/build-client#python


class MCPClient:
    def __init__(self, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.command = command
        self.args = args
        self.env = env or None

    async def init(self) -> None:
        """
            Connect to an MCP server
        """
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self._resolve_env(),
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        print(f"\nConnected to server with available_tools:\n{json.dumps(available_tools, indent=2)}")

    async def close(self):
        """
        Clean up resources
        """
        try:
            await self.exit_stack.aclose()
        except BaseException:
            # Ignore errors during cleanup (like CancelledError or RuntimeError from anyio)
            pass

    async def get_tools(self) -> List[dict]:
        """
        Retrieve the list of available tools from the MCP server
        用于从 MCP 服务器获取可用工具列表, 并将其转换为字典列表格式返回
        mcp返回的工具包含 name, description, inputSchema 等字段, 后续会被用于构建 LLM 的工具调用定义, 
        必须符合 OpenAI function-calling 的规范
        """
        if not self.session:
            raise RuntimeError("MCPClient not initialized")

        response = await self.session.list_tools()
        tools = [{
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.inputSchema
        } for tool in response.tools]
        return tools
    
    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """
        Call a tool on the MCP server
        """
        if not self.session:
            raise RuntimeError("MCPClient not initialized")
            
        #refer to https://modelcontextprotocol.io/docs/develop/build-client#calling-tools

        result = await self.session.call_tool(tool_name, arguments)
        return result.content

    def _resolve_env(self) -> Optional[Dict[str, str]]:
        """
        Resolve env values like "$FOO" or "${FOO}" from process env,
        so secrets stay in .env and never hard-coded in config.
        """
        if not self.env:
            return None

        resolved: Dict[str, str] = {}
        for key, value in self.env.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_key = value[2:-1]
                resolved_value = os.environ.get(env_key)
            elif isinstance(value, str) and value.startswith("$"):
                env_key = value[1:]
                resolved_value = os.environ.get(env_key)
            else:
                resolved_value = value
                env_key = None

            if resolved_value is None:
                missing = env_key or key
                raise RuntimeError(f"Environment variable '{missing}' is not set for MCP server '{self.command}'")
            resolved[key] = resolved_value

        return resolved
