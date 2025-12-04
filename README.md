# Jarvis (Just IronMan's Assistant as you know)

**A lightweight, transparent, and developer-first AI Agent framework.**

> "Stop fighting the framework. Start building the agent."

## 💡 Motivation

This project was born out of frustration with heavy, black-box frameworks like LangChain. We've all been there: a library update breaks your RAG pipeline, or you spend hours debugging a stack trace buried 10 layers deep in abstract classes.

**Jarvis is different.** It is "hand-crafted" from first principles to be:
*   **Transparent**: No hidden magic. You can see the `while` loop that runs your agent.
*   **Stable**: No breaking changes every week. You own the logic.
*   **Standardized**: Built on the **Model Context Protocol (MCP)**, not proprietary tool wrappers.

## ⚡ Core Architecture

Jarvis treats the **Agent** as the absolute core. It is a custom-built reasoning engine that we "arm" with modular capabilities:

1.  **The Brain (Agent)**: A pure Python implementation of the ReAct/Loop pattern. It handles reasoning, planning, and execution without heavy abstractions.
2.  **The Hands (MCP_CORE)**: Instead of hard-coding tools, Jarvis implements a native **MCP Client**. It connects to any standard MCP Server (Filesystem, Git, Fetch) or your own custom Python scripts.
3.  **The Memory (RAG)**: A controllable, lightweight RAG pipeline that ingests PDFs, CSVs, and Markdown, giving your agent grounded context without the bloat.

## 📦 Quick Start

1.  **Clone & Install**
    ```bash
    git clone https://github.com/yourusername/Jarvis.git
    pip install -r requirements.txt
    ```

2.  **Configure**
    Set your keys in `.env` and define your task in `config/user_config.json`.

3.  **Run**
    ```bash
    python main.py
    ```


## 🔮 To Be Update

- [ ] **Advanced Vector Stores**: Support for Milvus, Chroma, and faiss for long-term memory.
- [ ] **Flexible RAG Strategies**: Pluggable modules for Query Rewriting, Reranking, and Hybrid Search.
- [ ] **Multi-Format Support**: Native parsing for `.pdf`, `.csv`, `.docx`, and `.pptx`.
- [ ] **ReAct Optimization**: Enhanced fallback strategies for local SLMs (Small Language Models) and toggleable MCP tools.
- [ ] **Local Fine-tuning**: Pipeline for fine-tuning Ollama models on your own data to improve domain-specific performance.

## 📄 License

MIT License
