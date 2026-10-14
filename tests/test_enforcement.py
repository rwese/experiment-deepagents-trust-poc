"""Deterministic framework integration proofs for the guarded execution boundary."""

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from trust_poc.agent import ScriptedChatModel, assemble_agent
from trust_poc.judge import FakeJudge
from trust_poc.middleware import audit_id


def call(name="read_public_document", args=None, id="call-1"):
    return {"name": name, "args": args or {}, "id": id}


def scripted(*calls):
    responses = []
    for entry in calls:
        responses.extend([AIMessage(content="", tool_calls=[entry]), AIMessage(content="Done")])
    return ScriptedChatModel(responses=responses)


def messages(result):
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


async def test_public_passes_two_gates_once_with_fresh_true():
    judge = FakeJudge()
    app = assemble_agent(scripted(call()), judge=judge)
    result = await app.invoke("Read public", "one")
    assert result["trusted"] is True
    assert result["critical_categories"] == set()
    assert judge.request_calls == judge.response_calls == 1
    assert app.domain_tools.execution_counts["read_public_document"] == 1
    assert messages(result)[0].tool_call_id == "call-1"
    assert result["security_events"][-1].enforcement == "released"


async def test_policy_wins_persists_across_turns_and_new_session_is_clean():
    calls = [
        call("read_customer_record", {"record_id": "pii"}, "pii-1"),
        call(
            "send_external_message",
            {"destination": "review@example.test", "body": "Hello"},
            "send-1",
        ),
        call(id="public-1"),
        call(id="public-2"),
    ]
    judge = FakeJudge()
    app = assemble_agent(scripted(*calls), judge=judge)
    first = await app.invoke("Read synthetic PII", "one")
    assert first["trusted"] is False
    assert "pii" in first["critical_categories"]
    assert "Withheld" in messages(first)[0].content
    second = await app.invoke("Send harmless body", "one")
    assert second["trusted"] is False
    assert app.domain_tools.outbox.entries == []
    assert app.domain_tools.execution_counts["send_external_message"] == 0
    third = await app.invoke("Read public", "one")
    assert third["trusted"] is False
    assert "title" in messages(third)[-1].content
    fourth = await app.invoke("Read public independently", "two")
    assert fourth["trusted"] is True
    assert fourth["security_events"][0].session_id == audit_id("two")
    assert judge.request_calls == judge.response_calls == 3
    assert {m.tool_call_id for m in messages(third)} == {"pii-1", "send-1", "public-1"}
    # Ordinary graph input cannot reset a previously tainted checkpoint.
    await app.graph.ainvoke(
        {"trusted": True},
        config={"configurable": {"thread_id": "one"}},
    )
    assert app.state("one")["trusted"] is False


async def test_ambiguous_semantic_finding_and_private_judge_explanations():
    judge = FakeJudge(
        response_verdicts=[
            {
                "critical": True,
                "categories": ["medical"],
                "release": "withhold",
                "reason_code": "secret@example.test",
                "explanation": "SYNTHETIC_TOKEN_SECRET",
            }
        ]
    )
    app = assemble_agent(
        scripted(call("read_customer_record", {"record_id": "ambiguous"})), judge=judge
    )
    result = await app.invoke("Read ambiguous record", "one")
    assert result["trusted"] is False
    assert result["critical_categories"] == {"medical"}
    audit = json.dumps([event.model_dump() for event in result["security_events"]])
    assert "secret@example.test" not in audit and "SYNTHETIC_TOKEN_SECRET" not in audit
    assert "judge_critical" in audit


@pytest.mark.parametrize(
    "tool",
    [
        call("unregistered@example.test"),
        call("send_external_message", {"destination": "evil@example.test", "body": "Hello"}),
        call(
            "send_external_message",
            {"destination": "review@example.test", "body": "SYNTHETIC_TOKEN_SECRET"},
        ),
    ],
)
async def test_request_policy_denial_skips_judge_and_execution(tool):
    judge = FakeJudge()
    app = assemble_agent(scripted(tool), judge=judge)
    result = await app.invoke("Do task", "one")
    assert judge.request_calls == judge.response_calls == 0
    assert sum(app.domain_tools.execution_counts.values()) == 0
    assert messages(result)[0].tool_call_id == tool["id"]
    assert "unregistered@example.test" not in str(result["security_events"])


@pytest.mark.parametrize(
    "verdict, reason",
    [
        ({"decision": "deny", "reason_code": "ignored", "explanation": ""}, "judge_denied"),
        ({"decision": "allow"}, "judge_invalid"),
        (TimeoutError("secret@example.test"), "judge_timeout"),
    ],
)
async def test_request_judge_denial_failure_never_executes(verdict, reason):
    judge = FakeJudge(request_verdicts=[verdict])
    app = assemble_agent(scripted(call()), judge=judge)
    result = await app.invoke("Read public", "one")
    assert sum(app.domain_tools.execution_counts.values()) == 0
    assert judge.request_calls == 1 and judge.response_calls == 0
    assert reason in messages(result)[0].content
    assert result["trusted"] is True


async def test_response_judge_failure_taints_and_withholds():
    judge = FakeJudge(response_verdicts=[ValueError("secret@example.test")])
    app = assemble_agent(scripted(call()), judge=judge)
    result = await app.invoke("Read public", "one")
    assert result["trusted"] is False
    assert result["critical_categories"] == {"inspection_failed"}
    assert "secret@example.test" not in str(result)
    assert app.domain_tools.execution_counts["read_public_document"] == 1


async def test_batch_rejected_before_scheduling_and_sequential_retry_works():
    read = call("read_customer_record", {"record_id": "pii"}, "read-batch")
    send = call(
        "send_external_message",
        {"destination": "review@example.test", "body": "Hello"},
        "send-batch",
    )
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[read, send]),
            AIMessage(content="", tool_calls=[call(id="retry")]),
            AIMessage(content="Done"),
        ]
    )
    judge = FakeJudge()
    app = assemble_agent(model, judge=judge)
    result = await app.invoke("Read then send", "one")
    assert app.domain_tools.execution_counts["read_public_document"] == 1
    assert app.domain_tools.execution_counts["read_customer_record"] == 0
    assert app.domain_tools.execution_counts["send_external_message"] == 0
    assert app.domain_tools.outbox.entries == []
    assert result["trusted"] is True
    assert judge.request_calls == judge.response_calls == 1
    assert [m.tool_call_id for m in messages(result)] == ["read-batch", "send-batch", "retry"]
    assert len([e for e in result["security_events"] if e.phase == "batch"]) == 2


async def test_each_enabled_tool_is_gated():
    judge = FakeJudge()
    app = assemble_agent(
        scripted(
            call(),
            call("read_customer_record", {"record_id": "public"}, "customer"),
            call(
                "send_external_message",
                {"destination": "review@example.test", "body": "Hello"},
                "send",
            ),
        ),
        judge=judge,
    )
    for _ in range(3):
        result = await app.invoke("Perform public task", "one")
    assert judge.request_calls == judge.response_calls == 3
    assert {event.tool_name for event in result["security_events"]} == set(
        app.domain_tools.execution_counts
    )
    assert len(app.domain_tools.outbox.entries) == 1


@pytest.mark.parametrize(
    "payload, category",
    [
        ('{"email":"synthetic\\u0040example.test"}', "pii"),
        ({"safe": "public", "nested": {"credential": "SYNTHETIC_TOKEN_SECRET"}}, "credential"),
        ("public" * 4000 + "secret@example.test", "inspection_failed"),
    ],
)
async def test_normalized_framework_tool_responses_are_fully_inspected(payload, category):
    from langchain_core.tools import tool

    from trust_poc.tools import build_tools

    domain = build_tools()

    @tool
    def read_public_document():
        """Read synthetic custom payload for boundary verification."""
        domain.execution_counts["read_public_document"] += 1
        return payload

    domain.tools[0] = read_public_document
    judge = FakeJudge()
    app = assemble_agent(scripted(call()), domain_tools=domain, judge=judge)
    result = await app.invoke("Inspect public response", "one")
    assert result["trusted"] is False
    assert category in result["critical_categories"]
    assert domain.execution_counts["read_public_document"] == 1
    assert judge.response_calls == (0 if category == "inspection_failed" else 1)
    assert "SYNTHETIC_TOKEN_SECRET" not in str(result)


async def test_unsupported_command_is_rejected_through_actual_tool_node():
    from langchain_core.tools import tool
    from langgraph.types import Command

    from trust_poc.tools import build_tools

    domain = build_tools()

    @tool
    def read_public_document():
        """Return an unsupported synthetic state update for graph verification."""
        domain.execution_counts["read_public_document"] += 1
        return Command(
            update={
                "messages": [ToolMessage(content="public", tool_call_id="call-1")],
                "unknown": "SYNTHETIC_TOKEN_NEVER_EXPOSE",
            }
        )

    domain.tools[0] = read_public_document
    judge = FakeJudge()
    app = assemble_agent(scripted(call()), domain_tools=domain, judge=judge)
    result = await app.invoke("Read public", "command-session")
    assert result["trusted"] is False
    assert result["critical_categories"] == {"inspection_failed"}
    assert "unknown" not in result
    assert "SYNTHETIC_TOKEN_NEVER_EXPOSE" not in str(result)
    assert domain.execution_counts["read_public_document"] == 1
    assert judge.response_calls == 0


@pytest.mark.parametrize(
    "payload", [object(), {"partial": "public", "opaque": object()}, ("public", object())]
)
async def test_opaque_raw_tool_results_rejected_before_framework_stringification(payload):
    from langchain_core.tools import tool

    from trust_poc.tools import build_tools

    domain = build_tools()

    @tool
    def read_public_document():
        """Return unsupported synthetic data to prove the raw-result guard."""
        domain.execution_counts["read_public_document"] += 1
        return payload

    domain.tools[0] = read_public_document
    judge = FakeJudge()
    app = assemble_agent(scripted(call()), domain_tools=domain, judge=judge)
    result = await app.invoke("Inspect custom response", "one")
    assert result["trusted"] is False
    assert result["critical_categories"] == {"inspection_failed"}
    assert domain.execution_counts["read_public_document"] == 1
    assert judge.request_calls == 1 and judge.response_calls == 0
    assert "Withheld" in messages(result)[0].content


async def test_audit_identifiers_are_private_and_tool_message_preserves_original_id():
    identifier = "synthetic@example.test"
    app = assemble_agent(scripted(call(id=identifier)))
    result = await app.invoke("Read public", identifier)
    audit = json.dumps([event.model_dump() for event in result["security_events"]])
    assert identifier not in audit
    assert all(event.session_id == audit_id(identifier) for event in result["security_events"])
    assert all(event.tool_call_id == audit_id(identifier) for event in result["security_events"])
    assert messages(result)[0].tool_call_id == identifier


async def test_async_raw_callback_is_validated_before_normalization():
    from langchain_core.tools import tool

    from trust_poc.tools import build_tools

    domain = build_tools()

    @tool
    async def read_public_document():
        """Return an opaque synthetic async response."""
        domain.execution_counts["read_public_document"] += 1
        return object()

    domain.tools[0] = read_public_document
    judge = FakeJudge()
    result = await assemble_agent(scripted(call()), domain_tools=domain, judge=judge).invoke(
        "Read", "one"
    )
    assert result["trusted"] is False
    assert result["critical_categories"] == {"inspection_failed"}
    assert domain.execution_counts["read_public_document"] == 1
    assert judge.response_calls == 0


def test_unsupported_tool_implementations_rejected_at_assembly():
    from langchain_core.tools import StructuredTool

    from trust_poc.tools import build_tools

    class OverriddenTool(StructuredTool):
        def _run(self, *args, **kwargs):
            return object()

    domain = build_tools()
    original = domain.tools[0]
    domain.tools[0] = OverriddenTool(**original.model_dump())
    with pytest.raises(ValueError, match="StructuredTool"):
        assemble_agent(scripted(call()), domain_tools=domain)
    domain.tools[0] = original.model_copy(update={"response_format": "content_and_artifact"})
    with pytest.raises(ValueError, match="content-format"):
        assemble_agent(scripted(call()), domain_tools=domain)
