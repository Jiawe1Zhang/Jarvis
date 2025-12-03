from typing import Any
import asyncio
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server_transport
import pydantic

# 1. 创建 Server 实例
server = Server("my-custom-tools")

# 2. 定义你的 Python 函数
# 比如一个简单的加法工具，或者任何复杂的业务逻辑
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="add_numbers",
            description="Add two numbers together",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        ),
        types.Tool(
            name="get_weather",
            description="Get weather information for a city (Mock)",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        )
    ]

# 3. 处理工具调用
@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name == "add_numbers":
        if not arguments:
            raise ValueError("Missing arguments")
        a = arguments.get("a")
        b = arguments.get("b")
        result = a + b
        return [types.TextContent(type="text", text=f"The sum is {result}")]
    
    elif name == "get_weather":
        city = arguments.get("city", "Unknown")
        # 这里可以调用真实的 API 或者你的业务逻辑
        return [types.TextContent(type="text", text=f"The weather in {city} is Sunny, 25°C (Mock Data)")]

    raise ValueError(f"Unknown tool: {name}")

# 4. 运行 Server
async def main():
    # Run the server using stdin/stdout streams
    async with stdio_server_transport() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="my-custom-tools",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
