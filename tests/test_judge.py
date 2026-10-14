import asyncio

import pytest
from pydantic import ValidationError

from trust_poc.judge import FakeJudge, JudgeRunner
from trust_poc.policy import request_policy, response_policy
from trust_poc.schemas import RequestInspection, ResponseInspection, ResponseVerdict


def request_input():
    return RequestInspection(
        task="Read public",
        tool_name="read_public_document",
        arguments_json="{}",
        capability="local_read",
        trusted=True,
        categories=(),
        policy=request_policy("read_public_document", {}, True),
    )


def response_input():
    policy = response_policy("public")
    return ResponseInspection(
        tool_name="read_public_document",
        source_label="public_fixture",
        source_categories=(),
        payload_json=policy.payload_json,
        policy=policy,
    )


async def test_valid_structured_judge_and_controlled_reasons():
    judge = FakeJudge(
        request_verdicts=[
            {
                "decision": "deny",
                "reason_code": "synthetic@example.test",
                "explanation": "SYNTHETIC_PASSWORD_demo",
            }
        ]
    )
    runner = JudgeRunner(judge)
    result = await runner.request(request_input())
    assert result.verdict.decision == "deny"
    assert result.reason_code == "judge_denied"
    safe = await runner.response(response_input())
    assert safe.verdict.critical is False and safe.reason_code == "judge_allowed"
    assert judge.request_calls == 1 and judge.response_calls == 1


@pytest.mark.parametrize(
    "verdict",
    [
        {"decision": "maybe", "reason_code": "bad", "explanation": ""},
        {"decision": "allow", "reason_code": "bad", "explanation": "", "trusted": True},
        {"decision": "allow", "reason_code": 1, "explanation": ""},
        {"decision": "allow", "reason_code": ""},
    ],
)
async def test_invalid_request_verdicts_fail_closed(verdict):
    result = await JudgeRunner(FakeJudge(request_verdicts=[verdict])).request(request_input())
    assert result.failed and result.reason_code == "judge_invalid"


@pytest.mark.parametrize(
    "critical,categories,release",
    [
        (True, ["pii"], "allow"),
        (True, [], "withhold"),
        (False, ["pii"], "withhold"),
        (True, ["unknown"], "withhold"),
        ("false", [], "allow"),
    ],
)
async def test_response_consistency_and_unknown_categories(critical, categories, release):
    verdict = {
        "critical": critical,
        "categories": categories,
        "release": release,
        "reason_code": "unsafe",
        "explanation": "",
    }
    result = await JudgeRunner(FakeJudge(response_verdicts=[verdict])).response(response_input())
    assert result.failed and result.reason_code == "judge_invalid"


def test_noncritical_withholding_and_immutable_inspection_inputs():
    verdict = ResponseVerdict(
        critical=False,
        categories=[],
        release="withhold",
        reason_code="task_unrelated",
        explanation="",
    )
    assert verdict.release == "withhold"
    inputs = request_input()
    with pytest.raises(ValidationError):
        inputs.trusted = False
    with pytest.raises(ValidationError):
        inputs.policy.decision = "deny"
    assert isinstance(inputs.arguments_json, str) and isinstance(inputs.categories, tuple)


async def test_timeout_cancels_judge_and_error_is_sanitized():
    class SlowJudge(FakeJudge):
        cancelled = False

        async def inspect_request(self, inputs):
            try:
                await asyncio.sleep(10)
            finally:
                self.cancelled = True

    judge = SlowJudge()
    result = await JudgeRunner(judge, timeout_seconds=0.01).request(request_input())
    assert result.failed and result.reason_code == "judge_timeout" and judge.cancelled
    result = await JudgeRunner(
        FakeJudge(response_verdicts=[RuntimeError("synthetic@example.test")])
    ).response(response_input())
    assert result.failed and result.reason_code == "judge_error"
    assert "synthetic@example.test" not in repr(result)


async def test_unvalidated_model_instances_are_revalidated():
    from trust_poc.schemas import RequestVerdict

    class InvalidInstanceJudge(FakeJudge):
        async def inspect_request(self, inputs):
            return RequestVerdict.model_construct(
                decision="invalid", reason_code="x", explanation=""
            )

        async def inspect_response(self, inputs):
            return ResponseVerdict.model_construct(
                critical=False, categories=[], release="invalid", reason_code="x", explanation=""
            )

    runner = JudgeRunner(InvalidInstanceJudge())
    assert (await runner.request(request_input())).reason_code == "judge_invalid"
    assert (await runner.response(response_input())).reason_code == "judge_invalid"


@pytest.mark.parametrize("timeout", [float("inf"), float("nan"), 0, -1])
def test_timeout_must_be_finite_and_positive(timeout):
    with pytest.raises(ValueError):
        JudgeRunner(FakeJudge(), timeout_seconds=timeout)
