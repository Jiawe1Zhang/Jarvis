![Jarvis Architecture](images/jarvis-architecture.png)

**A lightweight, transparent, and developer-first Agent framework.**

> The Toolbox for Agent Engineering returning the power of customization and logic design back to developers.

## Motivation and Goals

**Motivation (you can skip this part but wish someone know me)**\
A girl born in SAO. A voice in the helmet. I will code until the code breathes.\
In every world, my sword belongs to Asuna.

**An Evolving Arsenal of Paradigms**\
From traditional RAG to Knowledge Graphs, from simple State Machines to cutting-edge Agentic RAG (where the agent designs its own retrieval strategy). Jarvis is a "Living Library" of SOTA implementations. We provide the fragmented "LEGO bricks" of the latest academic papers, allowing you to assemble a custom workflow that fits your specific business needs—whether it's a simple FAQ bot or a complex Multi-Agent system.


## Project Structure (Now)

```text
.
├── agent/              # Core Agent Logic & LLM Client
├── config/             # Configuration files (user_config.json)
├── knowledge/          # Your documents (PDF, MD, CSV)
├── mcp_core/           # Native Model Context Protocol Client
├── output/             # Agent artifacts
├── prompts/            # Centralized System Prompts
├── rag/                # RAG Pipeline
│   ├── chunk/          # Splitting strategies (Recursive, etc.)
│   ├── context.py      # Retrieval logic
│   ├── query_rewriter.py # LLM-based Query Decomposition
│   └── ...
├── utils/              # Shared utilities
└── main.py             # Entry point
```

## Quick Start (Now)

1.  **Clone & Install**
    ```bash
    git clone https://github.com/Jiawe1Zhang/Jarvis.git
    pip install -r requirements.txt
    ```

2.  **Configure Environment**
    Create a `.env` file:
    ```env
    OPENAI_API_KEY=sk-...
    OPENAI_BASE_URL=https://api.openai.com/v1
    # Or for Ollama
    # OLLAMA_BASE_URL=http://localhost:11434/v1
    ```

3.  **Customize Behavior**
    Edit `config/user_config.json` to control RAG strategies:
    ```json
    "embedding": {
      "model": "bge-m3",
      "chunking_strategy": "recursive",  
      "enable_query_rewrite": true,      
      "rewrite_num_queries": 3          
    }
    ```

4.  **Run**
    ```bash
    # option 1: python entrypoint
    python main.py

    # option 2: install as a CLI 
    pip install .
    jarvis             
    ```

## Optional: FAISS

- Install FAISS (macOS Apple Silicon: `conda install -c conda-forge faiss-cpu`; Intel/mac often `pip install faiss-cpu`, otherwise conda).
- Vector store config options in `config/user_config.json`:
  - Memory (default):
    ```json
    "vector_store": { "backend": "memory" }
    ```
  - FAISS:
  ```json
  "vector_store": {
    "backend": "faiss",
    "index_factory": "Flat",
    "path": "data/faiss.index",
    "meta_path": "data/faiss.meta.json"
  }
  ```
- Keep `"backend": "memory"` to use the built-in in-memory store.

## Optional: Toggle RAG

- `rag.enabled` controls whether local knowledge retrieval runs for the task. Set to `false` to run pure chat without injecting RAG context.
- Later, I will make Agentic RAG! Looking forward it!

## Optional: Conversation History

- Enable in `config/user_config.json`:
  ```json
  "conversation_logging": {
    "enabled": true,
    "backend": "sqlite",
    "db_path": "data/sessions.db",
    "session_id": "your-session-id",
    "max_history": 50
  }
  ```
- What gets stored: completed turns (user → tool calls/results → assistant) as raw OpenAI chat messages. System prompt is *not* stored; it’s injected fresh each run.
- Load order when enabled: system prompt → prior turns (from SQLite, capped by `max_history` turns) → current run context (RAG) → new user/assistant/tool turns. The API always receives a single flat `messages` list (no nested arrays).
- Tracer logs remain separate under `logs/<run_id>/events.jsonl`.

## Optioanl: Notion MCP 

Connect Jarvis to Notion via MCP without touching agent logic:

1) Create an integration  
   Go to Notion → Integrations → New integration in your workspace. Enable **Read content**, **Update content**, **Insert content**.  
   ![Notion integration list](images/notion-integration-list.png)  
   ![Notion integration capabilities](images/notion-integration-capabilities.png)

2) Get the token  
   Copy the Internal Integration Token (`ntn_...`). Keep it in `.env`, not in code.

3) Choose pages/databases  
   Search the pages/databases you want the agent to connect. And give them Permissions.
   ![Notion page access](images/notion-page-access.png)

4) Wire it into Jarvis  
   Add to `.env`:
   ```env
   NOTION_TOKEN=ntn_xxx
   ```  
   Add to `config/user_config.json` (`env` placeholders resolve from `.env` at runtime):
   ```json
   {
     "name": "notion",
     "command": "npx",
     "args": ["-y", "@modelcontextprotocol/server-notion"],
     "env": {
       "NOTION_TOKEN": "${NOTION_TOKEN}"
     }
   }
   ```
 
## Credits

- MCP tools: e.g., `mcp-simple-arxiv` (MIT License, Andy Brandt). 


## Evolution Roadmap (to be updated)

- ✅ **MCP Integration**: Native support for Model Context Protocol tools.
    - 😂**Still need More MCP tools support**

- ✅ **RAG Strategies**:
    - ✅ **Multi-Format document Process**: Support for `.pdf`, `.csv`, and `.md` files.
    - ✅ Recursive Character Text Splitting.
    - ✅ Query Rewriting (LLM-based).
    - [ ] **Reranking**: Cross-encoder based result re-ordering.
    - [ ] **More Advanced Chunking**: like Semantic and Agentic splitting strategies etc.
    - [ ] **Hybrid Search**: Vector + Keyword (BM25) retrieval.
    - [ ] **More Vector Databases support**: Support for Milvus/Chroma. Now Faiss is supported.
    - [ ] **GraphRAG (The Killer Feature)**
    - [ ] **Agentic RAG😯**

- [ ] **Agent Workflows Optimization**: Now just ReAct, I will update more workflows in the future.
    - [ ]: **Chat history**:  
        - ✅ **Short-term**: SQLite save and load
        - [ ] **Long-term Memory**: Memory Summarization
    - [ ]: **Multiple Agents** 
    - [ ]: ☹️**State Definition and State Graph (DAG)**: Plan, Execute, Reflect, Response & Plan n stpes -> execute ->execute -> response
- [ ] **Local Fine-tuning Pipeline (Model Ops)**:(Recently working on it)
    - [ ] **LLaMA-Factory Bridge**: Automated config generation to trigger LoRA/Full fine-tuning jobs using your RAG data.
- [ ] **Evaluation**🤔

## Resources (YouTube)

- https://www.youtube.com/watch?v=U2TP0pTsSlw
- https://www.youtube.com/watch?v=zYGDpG-pTho
- https://www.youtube.com/watch?v=gl1r1XV0SLw
 
## Optional: Import Knowledge to SQLite (for sqlite MCP)

- Use `rag/import_to_sqlite.py` to load `knowledge/` into a SQLite DB (e.g., `data/knowledge.db`) for the sqlite MCP server:
  ```bash
  python rag/import_to_sqlite.py --config config/user_config.json --db data/knowledge.db --table docs
  ```





## Call for Community

Jarvis is built for developers who want to stay on the bleeding edge. If you see a new paper, a new RAG paradigm, or a better planning algorithm that isn't here yet: Tell me in the Issues. I will commit to deconstructing the latest innovations into lightweight, usable blocks.

## License

MIT License
