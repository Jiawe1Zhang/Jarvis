"""
Level 3: LLM-based dispatcher for granular tool selection and RAG decision.
Uses SimpleLLMClient with a strict JSON response_format.
"""
import json
from typing import Any, Dict, List, Optional

from agent.llm_client import SimpleLLMClient
from utils.prompt_loader import load_prompt


PROMPT_FILE = "router_dispatch_system.md"

DEFAULT_SYSTEM_PROMPT = """
You are the Final Decision Maker for Jarvis.
You will receive a user query and a set of "Provisional Decisions" made by the keyword/semantic layers.
Your task is to review these decisions and output the Final Configuration.

Inputs you will see:
- Query: The user's text.
- Provisional RAG: True/False/None from previous layers.
- Provisional Tools: tool_sets from previous layers (e.g., ["CODING"] or []).
- Available MCP servers/tools: Full list of what can be loaded.

Your Logic:
1. Verify: Reject false positives (e.g., "python snake" should NOT trigger CODING).
2. Complete: If RAG is None, decide based on context. If tools are missing/irrelevant, clear or add as needed.

Output JSON:
{
    "requires_rag": boolean,
    "tool_sets": ["CODING"],   // empty [] if none
    "specific_tools": [],      // optional; empty if none
    "reasoning": "Short justification of RAG/tools choice."
}
"""


class LLMRouter:
    """
    LLM-based router; returns structured dispatch info or None.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[SimpleLLMClient] = None,
        prompt_file: str = PROMPT_FILE,
    ) -> None:
        self.client = client or SimpleLLMClient(model=model, base_url=base_url, api_key=api_key)
        # Try loading prompt from prompts directory; fall back to baked-in default if missing.
        try:
            self.system_prompt = load_prompt(prompt_file)
        except Exception as exc:
            print(f"LLMRouter prompt load failed ({prompt_file}): {exc}")
            self.system_prompt = DEFAULT_SYSTEM_PROMPT

    def classify(
        self,
        query: str,
        provisional_rag: Optional[bool] = None,
        provisional_tools: Optional[List[str]] = None,
        available_servers: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not query:
            return None
        try:
            payload = {
                "query": query,
                "provisional_rag": provisional_rag,
                "provisional_tools": provisional_tools,
                "available_servers": available_servers or [],
            }
            prompt_text = json.dumps(payload, ensure_ascii=False, indent=2)
            raw = self.client.generate(
                prompt=prompt_text,
                system_prompt=self.system_prompt,
                response_format={"type": "json_object"},
            )
            data: Dict[str, Any] = json.loads(raw)
            if not isinstance(data, dict):
                return None
            requires_rag = bool(data.get("requires_rag"))
            tool_sets = data.get("tool_sets") or data.get("tool_domains") or []
            specific_tools = data.get("specific_tools") or []
            reasoning = data.get("reasoning", "")
            return {
                "requires_rag": requires_rag,
                "tool_sets": list(tool_sets),
                "specific_tools": list(specific_tools),
                "reasoning": reasoning,
            }
        except Exception as exc:
            print(f"LLMRouter parse error: {exc}")
        return None
