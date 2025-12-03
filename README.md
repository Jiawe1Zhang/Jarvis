# Jarvis Agent — MCP + Multi-LLM + RAG

轻量、可扩展的 Agent 骨架：从零实现 MCP 客户端，接入多模型（云端/OpenAI 兼容或本地 Ollama），再叠加 RAG 检索。目录尽量扁平，只按功能分三块：`mcp/`、`rag/`、`prompts/`。

## 项目结构
```
main.py      # 入口：加载配置 → 初始化 MCP → 调 Agent
src/
  mcp/        # Agent 调度 & MCP 客户端
    agent.py
    llm_client.py
    mcp_client.py
  rag/        # 检索与向量
    embedding_retriever.py
    vector_store.py
    context.py
  prompts/    # 系统提示
  config/     # 配置文件（用户/示例/加载器）
  utils.py    # 日志等通用工具
```

## 快速开始
1) 安装依赖  
   ```bash
   pip install -r requirements.txt
   ```
2) 配置（用户可改）  
   - 复制 `src/config/user_config.example.json` 为 `src/config/user_config.json`，按需修改：  
     - 选择 LLM（云端或本地 Ollama）、模型名、base_url/api_key  
     - 选择嵌入模型与端点  
     - 声明知识库路径（glob）、输出目录  
     - 配置要启动的 MCP servers（命令/参数）  
   - `.env` 里只需放敏感项（如 API Key），但也可直接写在 config 中。
3) 运行示例  
   ```bash
   python main.py
   ```

## 流程概览
1) 入口 `main.py` 读取任务，初始化 MCP clients（如 fetch / filesystem）。  
2) RAG（可选）：对 `knowledge/` 下的文档嵌入、检索，拼接为上下文。  
3) Agent 调用 LLM（支持工具调用），若有 `tool_calls` 则通过 MCP client 调 server 工具，回填结果继续对话。  
4) 返回最终回复，可扩展为生成文件、图或代码。

### 配置暴露 vs 开发者控制
- 用户可调：`src/config/user_config.json`（模型/端点、向量库、知识库路径、要启用的 MCP servers、任务模板）。  
- 开发者控制：`src/prompts/presets.py` 中的系统提示、内部工作流约束（默认不暴露给最终用户）。

## TODO / Roadmap
- [ ] MCP 客户端健壮性：重连、超时、多 server session 管理
- [ ] LLM 抽象：云端/本地可插拔（vLLM/Ollama/HF 后端）
- [ ] RAG 后端：Faiss/Milvus，混合检索（稀疏+稠密）
- [ ] 数据摄入：PDF/CSV/Markdown/HTML 解析与智能切块
- [ ] 文本→图（Mermaid/DOT 渲染）与领域 DSL 代码生成（few-shot/RAG 校验）
- [ ] 评估与调优：RAG 评测、代码/图校验，LoRA/QLoRA 钩子
- [ ] 部署与观测：本地/离线方案，配置驱动管线，日志/指标/trace

## 配置思路（预留）
- `config/mcp_servers.yaml`：声明要启动的 MCP server（命令行/环境变量/工具白名单）。  
- `config/models.yaml`：对话/嵌入模型的名称、端点、超时等。  
- `config/tasks/*.yaml`：用户任务、使用的知识库、可用工具集。

## 参考
- MCP 文档：https://modelcontextprotocol.io/  
- RAG 概览：https://scriv.ai/guides/retrieval-augmented-generation-overview/  
- 向量库：https://faiss.ai/ ，https://milvus.io/  
