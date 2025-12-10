English | [中文](README.zh.md)

![Jarvis Architecture](images/jarvis-architecture.png)

**Jarvis: A terminal-first research/knowledge assistant for grad students**

Focus: Arxiv search → local parse → summarization → Notion sync. Intent-aware tool routing, optional RAG, persistent chat history, and minimal setup.

## Project Structure

```text
.
├── agent/              # Core agent logic & LLM client
│   └── router/         # Intent router (L1 proposer + L3 reviewer)
├── config/             # User configs + MCP server registry
├── knowledge/          # Your documents (PDF, MD, CSV)
├── mcp_core/           # Native MCP client
├── output/             # Agent artifacts
├── papers/             # Arxiv PDFs/MD from the MCP server
├── prompts/            # System prompts
├── rag/                # RAG pipeline
│   ├── chunk/          # Splitters
│   ├── context.py      # Retrieval logic
│   ├── query_rewriter.py
│   └── ...
├── utils/              # Shared utilities
└── main.py             # Entry point (interactive loop)
```

## Quick Start

1) Clone & install  
```bash
git clone https://github.com/Jiawe1Zhang/Jarvis.git
pip install -r requirements.txt
```

2) Environment  
```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
# OLLAMA_BASE_URL=http://localhost:11434/v1
NOTION_TOKEN=ntn_...
```

3) Configure  
- User config: `config/user_config.json` (or your own).  
- MCP registry: `config/mcp_servers.json` (domains/tools for routing).

Key switches:
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

4) Run (interactive loop)  
```bash
python main.py \
  --config config/user_config.json \
  --mcp-registry config/mcp_servers.json
```
You’ll be prompted for a query; routing decides whether to use local knowledge and which MCP servers to load. MCP connections are opened on first use and reused within the process.

## Notes

- `.gitignore` excludes logs/, output/, papers/, knowledge/, and vector index artifacts.  
- Commit `pyproject.toml`; commit `uv.lock` only if you use `uv` for deps.  
- Arxiv MCP saves both PDF and parsed `.md` by design.  
- If you renamed the GitHub repo, update your remote:  
  `git remote set-url origin https://github.com/<you>/jarvis.git`.
