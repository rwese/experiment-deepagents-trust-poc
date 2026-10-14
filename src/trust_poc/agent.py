"""Deep Agents assembly and checkpointed, offline session harness."""

import asyncio
from collections.abc import Sequence
from typing import Any

from deepagents import create_deep_agent
from deepagents.graph import DeepAgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from trust_poc.tools import DomainTools, build_tools


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

    def __init__(self, graph: Any, domain_tools: DomainTools) -> None:
        self.graph = graph
        self.domain_tools = domain_tools

    async def invoke(self, task: str, session_id: str) -> dict[str, Any]:
        if not session_id:
            raise ValueError("session_id must be nonempty")
        return await self.graph.ainvoke(
            {"messages": [HumanMessage(content=task)]},
            config={"configurable": {"thread_id": session_id}, "recursion_limit": 30},
        )

    def state(self, session_id: str) -> dict[str, Any]:
        return self.graph.get_state({"configurable": {"thread_id": session_id}}).values


def assemble_agent(
    model: BaseChatModel,
    *,
    domain_tools: DomainTools | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    state_schema: type[DeepAgentState] | None = None,
) -> SessionAgent:
    """Expose only three domain tools, with no filesystem, task, or summarization hooks."""
    domain = domain_tools if domain_tools is not None else build_tools()
    graph = create_deep_agent(
        model=model,
        tools=domain.tools,
        middleware=[
            DisabledBuiltin("FilesystemMiddleware"),
            DisabledBuiltin("SubAgentMiddleware"),
            DisabledBuiltin("SummarizationMiddleware"),
            *middleware,
        ],
        state_schema=state_schema,
        checkpointer=InMemorySaver(),
        system_prompt="Use only the supplied tools. Request one tool call at a time.",
    )
    return SessionAgent(graph, domain)


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
