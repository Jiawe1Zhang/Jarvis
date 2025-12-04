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
    Set your keys in `.env` and define your task in `src/config/user_config.json`.

3.  **Run**
    ```bash
    python -m src.main
    ```


## 🔮 To Be Update

*   **Advanced Vector Stores**: Support for large-scale vector databases (Milvus, Chroma, pgvector) to enable long-term agent memory and massive knowledge bases.
*   **Flexible RAG Strategies**: Pluggable modules for Query Rewriting, Reranking (Cohere/BGE-Reranker), and advanced chunking strategies.
*   **Multi-Format Support**: Native parsing for `.pdf`, `.csv`, and `.docx` files.
*   **ReAct Optimization**: Enhanced ReAct loop with robust fallback strategies for local LLMs (regex parsing for models with weak function calling) and toggleable MCP support.

## 📄 License

MIT License
