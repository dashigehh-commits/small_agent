# Minimal Agent Runtime

从零实现的 Python Agent 核心循环，不依赖 LangGraph/OpenHands 等框架。默认通过 DeepSeek 官方 OpenAI-compatible Chat Completions API 进行真实 LLM 调用（当前默认模型 `deepseek-chat`；也可切换 `deepseek-reasoner`）。

## 运行

```bash
pip install -r requirements.txt
set DEEPSEEK_API_KEY=你的key   # PowerShell: $env:DEEPSEEK_API_KEY="..."
python -c "import asyncio; from agent import Agent; print(asyncio.run(Agent().run('北京天气如何？')) )"
```

可设置 `DEEPSEEK_MODEL`、`DEEPSEEK_BASE_URL`；也兼容 `OPENAI_API_KEY`、`OPENAI_MODEL`、`OPENAI_BASE_URL` 覆盖默认值。默认提供 calculator（安全 AST 计算）、search（mock）、weather（彩云实时接口）和 todo 工具。配置 `CAIYUN_API_TOKEN` 后，weather 会先地理编码再请求彩云 API。

## 设计

`Agent.run` 追加用户消息，调用 LLM；若返回 `tool_calls`，解析 JSON 参数、执行注册工具并把 `role=tool` 结果放回上下文，继续循环；无工具调用则返回最终答案。每次执行返回 `trace`，记录工具、参数和结果。每个 `session_id` 对应独立消息列表，窗口间互不影响，可用同一 id 追问。

上下文超过 `max_messages` 时保留首条消息和最近消息，并插入摘要占位，避免无限增长；`max_rounds` 防止模型死循环。异常（未知工具、参数错误、API 错误、轮次超限）均抛出清晰错误，工具错误会作为结果回传给模型。

## 测试

```bash
pytest -q
```

测试覆盖工具注册、工具调用循环、trace 与 session 隔离。生产环境可将 mock search/weather 替换为真实服务。

## Prompt 与问题记录

系统消息可在业务层加入工具使用规范、回答语言和安全策略；本 runtime 保持中立，仅发送历史消息与工具 Schema。开发记录：优先保证可观测的四步循环、严格参数解析和会话隔离，再增加轻量压缩；复杂向量记忆不在最小实现范围。
