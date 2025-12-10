中文 | [English](README.md)

![Jarvis Architecture](images/jarvis-architecture.png)

**Jarvis：面向研究生的终端科研/知识管理助手**

聚焦链路：Arxiv 检索 → 本地解析/摘要 → Notion 同步。具备意图路由、工具精筛、可选 RAG、多轮会话记忆，设置简单。

## 目录结构

```text
.
├── agent/              # 核心逻辑 & LLM 客户端
│   └── router/         # 意图路由（L1 关键词提案 + L3 复核）
├── config/             # 用户配置 + MCP 服务器注册表
├── knowledge/          # 你的知识库（PDF/MD/CSV）
├── mcp_core/           # 原生 MCP 客户端
├── output/             # 代理产物
├── papers/             # Arxiv MCP 生成的 PDF/MD
├── prompts/            # 系统提示词
├── rag/                # RAG 流程
│   ├── chunk/          # 分段策略
│   ├── context.py      # 检索逻辑
│   ├── query_rewriter.py
│   └── ...
├── utils/              # 工具函数
└── main.py             # 入口（交互循环）
```

## 快速上手

1) 克隆与安装  
```bash
git clone https://github.com/Jiawe1Zhang/Jarvis.git
pip install -r requirements.txt
```

2) 环境变量  
```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
# OLLAMA_BASE_URL=http://localhost:11434/v1
NOTION_TOKEN=ntn_...
```

3) 配置  
- 用户配置：`config/user_config.json`（或自定义）。  
- MCP 注册表：`config/mcp_servers.json`（域/工具，用于路由筛选）。

关键开关示例：
```json
"knowledge": { "enabled": true },
"intent_router": { "enabled": true },
"conversation_logging": {
  "enabled": true,
  "db_path": "data/sessions.db",
  "session_id": "your-session-id",
  "max_history": 5
}
```

4) 运行（交互循环）  
```bash
python main.py \
  --config config/user_config.json \
  --mcp-registry config/mcp_servers.json
```
终端输入 query；路由决定是否用本地知识，以及加载哪些 MCP 服务器。MCP 连接首次按需建立，进程内复用。

## 其他说明

- `.gitignore` 已忽略 logs/、output/、papers/、knowledge/ 及向量索引产物。  
- `pyproject.toml` 建议提交；`uv.lock` 仅在用 `uv` 管理依赖时提交。  
- Arxiv MCP 默认会保存 PDF 和解析生成的 `.md`。  
- 如果你重命名了 GitHub 仓库，更新远端：  
  `git remote set-url origin https://github.com/<you>/jarvis.git`。
