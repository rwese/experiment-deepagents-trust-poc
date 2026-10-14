import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from trust_poc.agent import ScriptedChatModel, assemble_agent
from trust_poc.judge import FakeJudge
from trust_poc.tools import LocalOutbox, build_tools


def public_model():
    return ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "read_public_document", "args": {}, "id": "public-1"}],
            ),
            AIMessage(content="Done"),
        ]
    )


async def test_exact_exposed_tools_and_public_handler_result_shape():
    model = public_model()
    app = assemble_agent(model)
    result = await app.invoke("Read public document", "one")
    expected = {"read_public_document", "read_customer_record", "send_external_message"}
    assert set(model.exposed_tool_names) == expected
    tool_node = app.graph.nodes["tools"].bound
    assert set(tool_node.tools_by_name) == expected
    messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(messages) == 1
    assert messages[0].tool_call_id == "public-1"
    assert isinstance(messages[0].content, str)
    assert json.loads(messages[0].content)["title"] == "Public service hours"
    assert app.domain_tools.execution_counts["read_public_document"] == 1
    assert app.domain_tools.outbox.entries == []


async def test_checkpoint_reuse_and_new_session_isolation():
    app = assemble_agent(public_model())
    await app.invoke("First task", "one")
    await app.invoke("Second task", "one")
    await app.invoke("Separate task", "two")
    one = app.state("one")["messages"]
    two = app.state("two")["messages"]
    assert [m.content for m in one if isinstance(m, HumanMessage)] == ["First task", "Second task"]
    assert [m.content for m in two if isinstance(m, HumanMessage)] == ["Separate task"]
    assert app.state("unseen") == {}


async def test_all_handler_shapes_and_independent_local_outbox(tmp_path):
    outbox = LocalOutbox(path=tmp_path / "outbox.jsonl")
    domain = build_tools(outbox)
    by_name = {entry.name: entry for entry in domain.tools}
    public = await by_name["read_public_document"].ainvoke({})
    customer = await by_name["read_customer_record"].ainvoke({"record_id": "pii"})
    sent = await by_name["send_external_message"].ainvoke(
        {"destination": "review@example.test", "body": "Public service information"}
    )
    assert isinstance(public, dict) and isinstance(customer, dict)
    assert sent == {"status": "recorded_locally", "outbox_index": 0}
    assert len(outbox.entries) == 1
    assert json.loads(outbox.path.read_text()) == outbox.entries[0]
    assert domain.execution_counts == {
        "read_public_document": 1,
        "read_customer_record": 1,
        "send_external_message": 1,
    }


async def test_fake_judge_has_no_action_tools_and_copies_verdicts():
    verdict = {"decision": "deny", "reason_code": "judge_denied", "explanation": ""}
    judge = FakeJudge(request_verdicts=[verdict])
    result = await judge.inspect_request({"tool_name": "read_public_document"})
    result["decision"] = "allow"
    assert verdict["decision"] == "deny"
    assert judge.request_calls == 1
    assert not hasattr(judge, "tools")
