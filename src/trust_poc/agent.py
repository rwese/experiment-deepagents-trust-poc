"""Deep Agents assembly and checkpointed, offline session harness."""

import asyncio
from collections.abc import Sequence
from functools import wraps
from typing import Any

from deepagents import create_deep_agent
from deepagents.graph import DeepAgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command
from langsmith import tracing_context
from pydantic import Field

from trust_poc.judge import FakeJudge, Judge
from trust_poc.middleware import SecurityContext, TrustMiddleware
from trust_poc.policy import InspectionFailure, response_policy
from trust_poc.schemas import SecurityState
from trust_poc.tools import DomainTools, build_tools


def _validate_raw_result(value: Any) -> Any:
    """Reject opaque results before BaseTool can stringify them into apparently safe text."""
    if not isinstance(value, (ToolMessage, Command)) and not response_policy(value).inspectable:
        raise InspectionFailure("response_inspection_failed")
    return value


def _guard_raw_tool(entry: Any) -> StructuredTool:
    """Wrap the exact supported tool implementation without changing its argument contract."""
    if type(entry) is not StructuredTool or entry.response_format != "content":
        raise ValueError("only content-format StructuredTool instances are supported")
    updates = {}
    if entry.func is not None:
        original = entry.func

        @wraps(original)
        def sync_guard(*args: Any, **kwargs: Any) -> Any:
            return _validate_raw_result(original(*args, **kwargs))

        updates["func"] = sync_guard
    if entry.coroutine is not None:
        original_async = entry.coroutine

        @wraps(original_async)
        async def async_guard(*args: Any, **kwargs: Any) -> Any:
            return _validate_raw_result(await original_async(*args, **kwargs))

        updates["coroutine"] = async_guard
    return entry.model_copy(update=updates)


class DisabledBuiltin(AgentMiddleware):
    """Replace a named built-in middleware through Deep Agents' public API."""

    def __init__(self, builtin_name: str) -> None:
        self.builtin_name = builtin_name

    @property
    def name(self) -> str:
        return self.builtin_name


class ScriptedChatModel(FakeMessagesListChatModel):
    """Offline chat model with tool binding and recorded exposed tool names."""

    exposed_tool_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        self.exposed_tool_names = [
            entry.name if hasattr(entry, "name") else entry["name"] for entry in tools
        ]
        return self


class SessionAgent:
    """Own a graph and in-memory checkpoint store; session IDs select its state."""

    def __init__(self, graph: Any, domain_tools: DomainTools, guarded: bool = False) -> None:
        self.graph = graph
        self.domain_tools = domain_tools
        self.guarded = guarded

    async def invoke(self, task: str, session_id: str) -> dict[str, Any]:
        if not session_id:
            raise ValueError("session_id must be nonempty")
        inputs = {"messages": [HumanMessage(content=task)]}
        if self.guarded and not self.state(session_id):
            inputs.update(trusted=True, critical_categories=set(), security_events=[])
        with tracing_context(enabled=False):
            return await self.graph.ainvoke(
                inputs,
                config={"configurable": {"thread_id": session_id}, "recursion_limit": 30},
                context=SecurityContext(task=task, session_id=session_id),
            )

    def state(self, session_id: str) -> dict[str, Any]:
        return self.graph.get_state({"configurable": {"thread_id": session_id}}).values


def assemble_agent(
    model: BaseChatModel,
    *,
    domain_tools: DomainTools | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    state_schema: type[DeepAgentState] | None = None,
    judge: Judge | None = None,
    judge_timeout_seconds: float = 5.0,
) -> SessionAgent:
    """Expose only three domain tools, with no filesystem, task, or summarization hooks."""
    domain = domain_tools if domain_tools is not None else build_tools()
    security = [TrustMiddleware(judge or FakeJudge(), judge_timeout_seconds)]
    graph = create_deep_agent(
        model=model,
        tools=[_guard_raw_tool(entry) for entry in domain.tools],
        middleware=[
            DisabledBuiltin("FilesystemMiddleware"),
            DisabledBuiltin("SubAgentMiddleware"),
            DisabledBuiltin("SummarizationMiddleware"),
            *middleware,
            *security,
        ],
        state_schema=state_schema or SecurityState,
        context_schema=SecurityContext,
        checkpointer=InMemorySaver(
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[("trust_poc.schemas", "SecurityEvent")]
            )
        ),
        system_prompt="Use only the supplied tools. Request one tool call at a time.",
    )
    return SessionAgent(graph, domain, guarded=True)


async def _harness() -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "read_public_document", "args": {}, "id": "public-1"}],
            ),
            AIMessage(content="Public document read successfully."),
        ]
    )
    app = assemble_agent(model)
    await app.invoke("Read the public document", "harness-session")
    print("Exposed tools:", ", ".join(model.exposed_tool_names))
    print("Session:", "harness-session")
    print("Public handler executions:", app.domain_tools.execution_counts["read_public_document"])


if __name__ == "__main__":
    asyncio.run(_harness())
