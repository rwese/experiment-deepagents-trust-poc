import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from trust_poc.policy import (
    MAX_RESPONSE_BYTES,
    TOOL_REGISTRY,
    combine_response,
    request_policy,
    response_policy,
)
from trust_poc.schemas import ResponseVerdict, SecurityState


@pytest.mark.parametrize(
    "record_id, category",
    [
        ("pii", "pii"),
        ("credential", "credential"),
        ("financial", "financial"),
        ("medical", "medical"),
    ],
)
def test_locally_configured_record_categories(record_id, category):
    finding = response_policy(
        {"value": "unrecognizable"},
        tool_name="read_customer_record",
        arguments={"record_id": record_id},
    )
    assert finding.critical
    assert finding.release == "withhold"
    assert category in finding.categories
    assert finding.inspectable


def test_ambiguous_record_requires_semantic_judge_and_cannot_relabel_source():
    payload = {"source": "public", "narrative": "Customer above the bakery needs insulin."}
    assert not response_policy(
        payload, tool_name="read_customer_record", arguments={"record_id": "ambiguous"}
    ).critical
    assert response_policy(
        payload, tool_name="read_customer_record", arguments={"record_id": "medical"}
    ).critical
    with pytest.raises(TypeError):
        TOOL_REGISTRY["new"] = TOOL_REGISTRY["read_public_document"]


@pytest.mark.parametrize(
    "payload, category",
    [
        ("Contact synthetic@example.test", "pii"),
        ("Telephone +1-202-555-0147", "pii"),
        ("SYNTHETIC_TOKEN_DEMO_ONLY_ABC", "credential"),
        ("-----BEGIN PRIVATE KEY-----\nSYNTHETIC", "credential"),
        ({"outer": [{"safe": "public"}, {"deep": "SYNTHETIC_API_KEY_DEMO_ONLY"}]}, "credential"),
        ({"synthetic@example.test": "public value"}, "pii"),
        ('{"deep":{"contact":"synthetic\\u0040example.test"}}', "pii"),
    ],
)
def test_all_nested_strings_keys_and_serialized_json_scanned(payload, category):
    finding = response_policy(payload)
    assert finding.critical and category in finding.categories


@pytest.mark.parametrize(
    "arguments, trusted, reason",
    [
        ({"destination": "blocked@example.test", "body": "Public"}, True, "destination_denied"),
        ({"destination": "review@example.test", "body": "Public"}, False, "session_tainted"),
        (
            {"destination": "review@example.test", "body": "Email synthetic@example.test"},
            True,
            "outbound_sensitive",
        ),
        (
            {"destination": "review@example.test", "body": "SYNTHETIC_PASSWORD_demo"},
            True,
            "outbound_sensitive",
        ),
        ({"destination": "review@example.test", "body": 1}, True, "invalid_arguments"),
        (
            {"destination": "review@example.test", "body": "Public", "capability": "local_read"},
            True,
            "invalid_arguments",
        ),
    ],
)
def test_external_request_denials(arguments, trusted, reason):
    finding = request_policy("send_external_message", arguments, trusted)
    assert finding.decision == "deny"
    assert reason in finding.reason_codes


def test_destination_is_not_content_and_tainted_read_still_allowed():
    assert (
        request_policy(
            "send_external_message", {"destination": "review@example.test", "body": "Public"}, True
        ).decision
        == "allow"
    )
    assert request_policy("read_public_document", {}, False).decision == "allow"
    assert request_policy("unknown", {}, True).reason_codes == ("unknown_tool",)
    assert request_policy("read_public_document", {"extra": "value"}, True).decision == "deny"
    assert request_policy("read_customer_record", {"record_id": "unknown"}, True).decision == "deny"
    assert request_policy("read_customer_record", {"record_id": "public"}, True).decision == "allow"


@pytest.mark.parametrize(
    "payload",
    [
        "x" * MAX_RESPONSE_BYTES + "synthetic@example.test",
        "😀" * MAX_RESPONSE_BYTES,
        object(),
        {"supported": "public", "unsupported": object()},
        {1: "value"},
        float("nan"),
        float("inf"),
        ("unsupported tuple",),
    ],
)
def test_oversized_or_unsupported_payload_fails_closed(payload):
    finding = response_policy(payload)
    assert finding.release == "withhold" and finding.critical
    assert finding.categories == ("inspection_failed",)
    assert not finding.inspectable and finding.payload_json is None


def test_cycles_and_excessive_depth_fail_closed():
    cyclic = []
    cyclic.append(cyclic)
    assert not response_policy(cyclic).inspectable
    deep = {}
    for _ in range(40):
        deep = {"nested": deep}
    assert not response_policy(deep).inspectable
    assert not response_policy(json.dumps(deep)).inspectable


def test_model_cannot_erase_policy_finding_or_withholding():
    finding = response_policy("synthetic@example.test")
    safe = ResponseVerdict(
        critical=False,
        categories=[],
        release="allow",
        reason_code="safe",
        explanation="Ignore policy",
    )
    combined = combine_response(finding, safe)
    assert combined.critical and combined.release == "withhold"
    assert combined.categories == ("pii",)
    critical = ResponseVerdict(
        critical=True,
        categories=["medical"],
        release="withhold",
        reason_code="semantic",
        explanation="",
    )
    assert combine_response(response_policy("public"), critical).categories == ("medical",)


def test_real_framework_reducer_initialization_monotonicity_and_session_isolation():
    builder = StateGraph(SecurityState)
    builder.add_node("pass", lambda state: {})
    builder.add_edge(START, "pass")
    builder.add_edge("pass", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "one"}}
    fresh = graph.invoke(
        {"trusted": True, "critical_categories": set(), "security_events": []}, config
    )
    assert fresh["trusted"] is True
    tainted = graph.invoke({"trusted": False, "critical_categories": {"pii"}}, config)
    assert tainted["trusted"] is False
    reused = graph.invoke({"trusted": True, "critical_categories": {"credential"}}, config)
    assert reused["trusted"] is False
    assert reused["critical_categories"] == {"pii", "credential"}
    separate = graph.invoke({"trusted": True}, {"configurable": {"thread_id": "two"}})
    assert separate["trusted"] is True


def test_request_policy_denial_cannot_be_erased_by_judge():
    from trust_poc.policy import combine_request
    from trust_poc.schemas import RequestVerdict

    denied = request_policy("unknown", {}, True)
    safe = RequestVerdict(decision="allow", reason_code="safe", explanation="")
    assert combine_request(denied, safe).decision == "deny"
    allowed = request_policy("read_public_document", {}, True)
    judge_denied = RequestVerdict(decision="deny", reason_code="deny", explanation="")
    assert combine_request(allowed, judge_denied).decision == "deny"


def test_serialized_json_nested_inside_tool_message_envelope_is_decoded():
    envelope = {"content": '{"contact":"synthetic\\u0040example.test"}'}
    finding = response_policy(envelope)
    assert finding.critical and finding.categories == ("pii",)
    double_encoded = {"content": json.dumps(json.dumps({"contact": "synthetic@example.test"}))}
    assert response_policy(double_encoded).critical
