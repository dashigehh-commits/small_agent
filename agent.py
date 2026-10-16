"""A tiny, framework-free agent runtime with tool calling and sessions."""

from __future__ import annotations
import ast
import json
import operator
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable
import httpx
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT

load_dotenv()


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    function: Callable[..., Any]

    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    # 注册表是 Agent 与具体工具之间的唯一耦合点，便于增删工具。
    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def schemas(self):
        return [t.schema() for t in self.tools.values()]

    def call(self, name, args):
        if name not in self.tools:
            raise ValueError(f"unknown tool: {name}")
        return self.tools[name].function(**args)


def calculator(expression: str):
    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in allowed:
            return allowed[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in allowed:
            return allowed[type(n.op)](ev(n.operand))
        raise ValueError("unsafe expression")

    return ev(ast.parse(expression, mode="eval").body)


def search(query: str):
    return {"query": query, "results": [f"Mock result for: {query}"]}


def todo(task: str):
    return {"task": task, "status": "added"}


def weather(city: str):
    token = os.getenv("CAIYUN_API_TOKEN") or os.getenv("CAIYUN_TOKEN")
    if not token:
        raise RuntimeError("CAIYUN_API_TOKEN is not configured")
    geo = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "zh", "format": "json"},
        timeout=15,
    )
    geo.raise_for_status()
    results = geo.json().get("results") or []
    if not results:
        raise ValueError(f"city not found: {city}")
    lat, lon = results[0]["latitude"], results[0]["longitude"]
    r = httpx.get(
        f"https://api.caiyunapp.com/v2.6/{token}/{lon},{lat}/realtime", timeout=15
    )
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError(data.get("status", "Caiyun API error"))
    x = data.get("result", {}).get("realtime", {})
    return {
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "temperature_c": x.get("temperature"),
        "humidity": x.get("humidity"),
        "skycon": x.get("skycon"),
        "wind": x.get("wind"),
        "source": "caiyun",
    }


class LLMClient:
    def __init__(self, model=None, base_url=None, api_key=None):
        # DeepSeek exposes an OpenAI-compatible endpoint. Environment variables
        # remain overridable for other compatible providers.
        self.model = model or os.getenv(
            "DEEPSEEK_MODEL", os.getenv("OPENAI_MODEL", "deepseek-chat")
        )
        self.base = base_url or os.getenv(
            "DEEPSEEK_BASE_URL",
            os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        )
        self.key = api_key or os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY"))

    async def complete(self, messages, tools):
        if not self.key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not configured (or set OPENAI_API_KEY)"
            )
        system = {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
        request_messages = [system] + messages
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                self.base.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"},
                json={
                    "model": self.model,
                    "messages": request_messages,
                    "tools": tools,
                    "tool_choice": "auto",
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]


@dataclass
class Session:
    id: str
    messages: list[dict] = field(default_factory=list)


class Agent:
    def __init__(self, llm=None, max_rounds=8, max_messages=30):
        self.llm = llm or LLMClient()
        self.max_rounds = max_rounds
        self.max_messages = max_messages
        self.sessions = {}
        self.registry = ToolRegistry()
        for n, d, p, f in [
            (
                "calculator",
                "Evaluate arithmetic",
                '{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}',
                calculator,
            ),
            (
                "search",
                "Search web (mock)",
                '{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}',
                search,
            ),
            (
                "todo",
                "Add a todo",
                '{"type":"object","properties":{"task":{"type":"string"}},"required":["task"]}',
                todo,
            ),
            (
                "weather",
                "Get weather (mock)",
                '{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}',
                weather,
            ),
        ]:
            self.registry.register(Tool(n, d, json.loads(p), f))

    def session(self, sid=None):
        sid = sid or str(uuid.uuid4())
        return self.sessions.setdefault(sid, Session(sid))

    def _compact(self, s):
        # 轻量压缩：保留首条上下文和最近消息，避免长期对话无限增长。
        if len(s.messages) > self.max_messages:
            s.messages = [
                s.messages[0],
                {
                    "role": "system",
                    "content": "Earlier conversation summarized; retain user preferences and tool results.",
                },
            ] + s.messages[-self.max_messages + 2 :]

    async def run(self, user_input, sid=None):
        # 每个 session 拥有独立消息列表，因此不同窗口不会互相污染。
        s = self.session(sid)
        s.messages.append({"role": "user", "content": user_input})
        trace = []
        for _ in range(self.max_rounds):
            self._compact(s)
            msg = await self.llm.complete(s.messages, self.registry.schemas())
            if msg.get("tool_calls"):
                # 将模型消息和每个工具结果写回上下文，再让模型决定下一步。
                s.messages.append(msg)
                for tc in msg["tool_calls"]:
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                        out = self.registry.call(tc["function"]["name"], args)
                        trace.append(
                            {
                                "tool": tc["function"]["name"],
                                "args": args,
                                "result": out,
                            }
                        )
                    except Exception as e:
                        out = {"error": str(e)}
                    s.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps(out),
                        }
                    )
                continue
            # 没有工具调用即视为最终答案，同时保留到历史供后续追问。
            answer = msg.get("content", "")
            s.messages.append({"role": "assistant", "content": answer})
            return {"session_id": s.id, "answer": answer, "trace": trace}
        raise RuntimeError("maximum agent rounds exceeded")

if __name__ == '__main__':
    import asyncio
    from agent import Agent


    async def main():
        agent = Agent()

        first = await agent.run("北京天气怎么样？", "window-1")
        print(first["answer"])

        second = await agent.run("那明天呢？", "window-1")
        print(second["answer"])


    asyncio.run(main())
