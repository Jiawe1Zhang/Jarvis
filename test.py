import asyncio

from src.MCP_Client import MCPClient


async def main() -> None:
    """
    使用 uvx 拉起官方 fetch MCP server，初始化后列出工具。
    需要本机有 uvx（uv）并可下载/运行 mcp-server-fetch。
    """
    client = MCPClient()
    try:
        await client.init("mcp-server-fetch", command="uvx")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
