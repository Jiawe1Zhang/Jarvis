import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from utils import ToolCall, log_title
from utils.ui import BaseUI
from utils.tracer import RunTracer
from utils.session_store import SessionStore


class ChatOpenAI:
    """
    Minimal wrapper around the OpenAI Chat Completions API with tool support.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        context: str = "",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tracer: Optional[RunTracer] = None,
        session_store: Optional[SessionStore] = None,
        session_id: Optional[str] = None,
        max_history_turns: Optional[int] = None,
        ui: Optional[BaseUI] = None,
    ) -> None:
        # Support either cloud (OpenAI-compatible) or local (e.g., Ollama) endpoints.
        resolved_base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OLLAMA_BASE_URL")
        )
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key and resolved_base_url:
            resolved_api_key = "ollama"
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY must be set (or provide api_key manually)")
        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
        )
        self.model = model
        self.messages: List[Dict[str, Any]] = []
        self.tools = tools or []
        self.tracer = tracer
        self.session_store = session_store
        self.session_id = session_id or "default"
        self.max_history_turns = max_history_turns if max_history_turns and max_history_turns > 0 else None
        self.ui = ui or BaseUI()
        # buffer for current turn
        self._pending_turn: List[Dict[str, Any]] = []
        # system prompt first
        # 消息最开始一定是 system prompt
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        # preload history turns (after system, before new context); each turn is a list of messages
        # 给消息列表先加载历史对话
        if self.session_store:
            turns = self.session_store.load_turns(self.session_id, limit=self.max_history_turns or 0)
            for turn in turns:
                self.messages.extend(turn)
        # current run context (e.g., RAG)
        # 消息列表再加入rag的上下文
        if context:
            self.messages.append({"role": "user", "content": context})

    def chat(self, prompt: Optional[str] = None) -> Dict[str, Any]:
        if not self.ui.enabled:
            log_title("CHAT")
        if prompt:
            user_msg = {"role": "user", "content": prompt}
            # 消息列表最后加入用户的 prompt
            self.messages.append(user_msg)

            # 这里是把当前用户消息也放到 pending_turn 里，等会儿 flush_history 的时候会一起存储
            # 这是为了保证用户的消息和后续模型的回复、工具调用结果都能作为一个完整的对话轮次存储下来
            # 历史消息里, 统一存储用户消息, 模型回复, 工具调用结果, 而不存储前面code里system prompt和rag上下文
            # 这里先加入用户query
            self._pending_turn.append(user_msg)

        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            # tools 是用来定义可用的工具, 
            tools=self._get_tools_definition() or None,
        )

        choice = response.choices[0].message
        content = choice.content or ""
        tool_calls = [
            ToolCall(
                id=tool_call.id or "",
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
            for tool_call in choice.tool_calls or []
        ]

        if not self.ui.enabled:
            log_title("RESPONSE")
            if content:
                print(content)

        if self.tracer:
            self.tracer.log_event(
                {
                    "type": "llm_call",
                    "model": self.model,
                    "messages": self.messages,
                    "tool_calls": [tc.__dict__ for tc in tool_calls],
                    "response_content": content,
                }
            )

        # 👇是为了在后续的对话中保留上下文和工具调用结果
        assistant_message: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            # tool_calls 是 OpenAI Chat Completions 响应里暴露的一个字段，用来承载模型生成的“函数调用”指令，类似 role/content 那样是协议的一部分
            # 当你在请求中提供 tools（function-calling 定义）时，模型如果决定调用工具，会在返回的 choices[0].message.tool_calls 里给出调用清单
            # 每个 tool_call 包含 id、type: "function"、function.name 和 function.arguments（字符串形式）。
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in tool_calls
            ]

        # 消息列表里加入, assistant 的回复,主要内容是 content 和 tool_calls, eg:
        # {"role": "assistant", "content": "...", "tool_calls": [...]}
        # but一般来说content 可能是空的, 因为模型可能直接调用工具而不输出文本
        # 目前看来, 输出content的时候一般是没有tool_calls的, 也差不多就是结束loop了
        self.messages.append(assistant_message)
        # 这条消息也放到 pending_turn 里，等会儿 flush_history 的时候会一起存储
        self._pending_turn.append(assistant_message)

        return {"content": content, "tool_calls": tool_calls}

    # 这个函数用于将工具调用的结果附加到消息列表中
    def append_tool_result(self, tool_call_id: str, tool_output: str) -> None:
        tool_msg = {
            "role": "tool",
            "content": tool_output,
            "tool_call_id": tool_call_id,
        }

        self.messages.append(tool_msg)
    
        # 这里加入工具调用结果, 准备放到历史里
        self._pending_turn.append(tool_msg)

    def flush_history(self) -> None:
        """
        Persist current turn when explicitly called (e.g., after a successful run).
        """
        if self.session_store and self._pending_turn:
            self.session_store.append_turn(
                self.session_id,
                self._pending_turn,
                max_turns=self.max_history_turns or 0,
            )
            self._pending_turn.clear()

    # 这个函数用于将工具列表转换为OpenAI API所需的格式
    def _get_tools_definition(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {}),
                },
            }
            for tool in self.tools
        ]


class SimpleLLMClient:
    """
    A stateless, lightweight LLM client for single-turn tasks like Query Rewriting.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        resolved_base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OLLAMA_BASE_URL")
        )
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key and resolved_base_url:
            resolved_api_key = "ollama"

        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
        )
        self.model = model

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            request_args = {"model": self.model, "messages": messages}
            if response_format is not None:
                request_args["response_format"] = response_format
            response = self.client.chat.completions.create(**request_args)
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"SimpleLLMClient error: {e}")
            return ""
