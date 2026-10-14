"""Sequential tool-boundary enforcement with checkpointed sensitivity state."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from trust_poc.judge import Judge, JudgeRunner
from trust_poc.policy import (
    TOOL_REGISTRY,
    combine_response,
    request_policy,
    response_policy,
    source_categories,
)
from trust_poc.schemas import (
    RequestInspection,
    ResponseFinding,
    ResponseInspection,
    SecurityEvent,
    SecurityState,
)


def audit_id(value: str) -> str:
    """Project untrusted identifiers into stable references without revealing their text."""
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SecurityContext:
    task: str
    session_id: str


def inspect_handler_result(result: Any, call_id: str) -> tuple[ToolMessage, Any]:
    """Accept only bounded message payloads and message-only, local commands."""
    if isinstance(result, Command):
        if result.graph is not None or result.goto or result.resume is not None:
            raise ValueError("unsupported_command")
        if type(result.update) is not dict or set(result.update) != {"messages"}:
            raise ValueError("unsupported_command")
        messages = result.update["messages"]
        if type(messages) is not list or len(messages) != 1:
            raise ValueError("unsupported_messages")
        result = messages[0]
    if not isinstance(result, ToolMessage) or result.tool_call_id != call_id:
        raise ValueError("unsupported_message")
    # Include every field that can carry data into the agent or checkpoint.
    payload = result.model_dump(mode="python")
    return result, payload


class TrustMiddleware(AgentMiddleware):
    state_schema = SecurityState

    def __init__(self, judge: Judge, timeout_seconds: float = 5.0) -> None:
        self.runner = JudgeRunner(judge, timeout_seconds)

    @staticmethod
    def _event(context: SecurityContext, call: dict, trusted: bool, **fields: Any) -> SecurityEvent:
        return SecurityEvent(
            session_id=audit_id(context.session_id),
            tool_call_id=audit_id(call["id"]),
            tool_name=call["name"] if call["name"] in TOOL_REGISTRY else "unknown_tool",
            trusted_before=trusted,
            **fields,
        )

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: SecurityState, runtime: Any) -> dict | None:
        message = state["messages"][-1]
        if not isinstance(message, AIMessage) or len(message.tool_calls) < 2:
            return None
        context = runtime.context
        trusted = state.get("trusted") is not False
        events = [
            self._event(
                context,
                call,
                trusted,
                phase="batch",
                policy_outcome="deny",
                judge_outcome="not_run",
                reason_codes=("batch_rejected",),
                categories=(),
                trusted_after=trusted,
                enforcement="batch_rejected",
            )
            for call in message.tool_calls
        ]
        return {
            "messages": [
                ToolMessage(
                    content="Denied: batch_rejected. Retry one tool call at a time.",
                    tool_call_id=call["id"],
                )
                for call in message.tool_calls
            ],
            "security_events": events,
            "jump_to": "model",
        }

    async def awrap_tool_call(self, request: Any, handler: Any) -> Command:
        call = request.tool_call
        context = request.runtime.context
        if not isinstance(context, SecurityContext) or (
            request.runtime.config.get("configurable", {}).get("thread_id") != context.session_id
        ):
            raise ValueError("session_context_mismatch")
        trusted = request.state.get("trusted") is not False
        policy = request_policy(call["name"], call["args"], trusted)
        judge_outcome = "not_run"
        reason_codes = policy.reason_codes
        denied = policy.decision == "deny"
        if not denied:
            verdict = await self.runner.request(
                RequestInspection(
                    task=context.task,
                    tool_name=call["name"],
                    arguments_json=json.dumps(call["args"], ensure_ascii=False),
                    capability=TOOL_REGISTRY[call["name"]].capability,
                    trusted=trusted,
                    categories=tuple(sorted(request.state.get("critical_categories", set()))),
                    policy=policy,
                )
            )
            reason_codes += (verdict.reason_code,)
            judge_outcome = "failed" if verdict.failed else verdict.verdict.decision
            denied = verdict.failed or verdict.verdict.decision == "deny"
        request_event = self._event(
            context,
            call,
            trusted,
            phase="request",
            policy_outcome=policy.decision,
            judge_outcome=judge_outcome,
            reason_codes=reason_codes,
            categories=policy.categories,
            trusted_after=trusted,
            enforcement="denied" if denied else "executed",
        )
        if denied:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="Denied: " + ", ".join(reason_codes), tool_call_id=call["id"]
                        )
                    ],
                    "security_events": [request_event],
                }
            )
        try:
            result = await handler(request)
            message, payload = inspect_handler_result(result, call["id"])
            finding = response_policy(payload, tool_name=call["name"], arguments=call["args"])
        except GraphBubbleUp:
            raise
        except Exception:  # noqa: BLE001 - handler/adapter failures withhold raw exception text
            message = None
            finding = ResponseFinding(
                release="withhold",
                critical=True,
                categories=("inspection_failed",),
                reason_codes=("response_inspection_failed",),
                inspectable=False,
            )
        combined = finding
        judge_outcome = "not_run"
        reason_codes = finding.reason_codes
        if finding.inspectable:
            verdict = await self.runner.response(
                ResponseInspection(
                    tool_name=call["name"],
                    source_label=TOOL_REGISTRY[call["name"]].source_label,
                    source_categories=source_categories(call["name"], call["args"]),
                    payload_json=finding.payload_json,
                    policy=finding,
                )
            )
            reason_codes += (verdict.reason_code,)
            if verdict.failed:
                judge_outcome = "failed"
                combined = ResponseFinding(
                    release="withhold",
                    critical=True,
                    categories=tuple(sorted(set(finding.categories) | {"inspection_failed"})),
                    reason_codes=reason_codes,
                    inspectable=False,
                )
            else:
                judge_outcome = verdict.verdict.release
                combined = combine_response(finding, verdict.verdict)
        trusted_after = trusted and not combined.critical
        response_event = self._event(
            context,
            call,
            trusted,
            phase="response",
            policy_outcome=finding.release,
            judge_outcome=judge_outcome,
            reason_codes=reason_codes,
            categories=combined.categories,
            trusted_after=trusted_after,
            enforcement="released" if combined.release == "allow" else "withheld",
        )
        if combined.release == "withhold":
            message = ToolMessage(
                content="Withheld: "
                + ", ".join(combined.categories)
                + "; "
                + ", ".join(reason_codes),
                tool_call_id=call["id"],
            )
        return Command(
            update={
                "messages": [message],
                "trusted": trusted_after,
                "critical_categories": set(combined.categories),
                "security_events": [request_event, response_event],
            }
        )
