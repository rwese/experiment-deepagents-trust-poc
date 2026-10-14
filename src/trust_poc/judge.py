"""Isolated asynchronous judge interface for the executable harness."""

# Inspection inputs are immutable contracts; raw provider verdicts are validated here.
import asyncio
import math
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from trust_poc.schemas import (
    RequestInspection,
    RequestVerdict,
    ResponseInspection,
    ResponseVerdict,
)


class Judge(Protocol):
    async def inspect_request(self, inputs: RequestInspection) -> Mapping[str, Any]: ...

    async def inspect_response(self, inputs: ResponseInspection) -> Mapping[str, Any]: ...


class FakeJudge:
    """Scripted verdicts and exceptions for offline tests; no action tools."""

    def __init__(
        self,
        request_verdicts: list[Mapping[str, Any] | Exception] | None = None,
        response_verdicts: list[Mapping[str, Any] | Exception] | None = None,
    ) -> None:
        self.request_verdicts = deque(request_verdicts or [])
        self.response_verdicts = deque(response_verdicts or [])
        self.request_calls = 0
        self.response_calls = 0

    async def inspect_request(
        self, inputs: RequestInspection | Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.request_calls += 1
        return self._next(
            self.request_verdicts,
            {"decision": "allow", "reason_code": "judge_allowed", "explanation": ""},
        )

    async def inspect_response(
        self, inputs: ResponseInspection | Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.response_calls += 1
        return self._next(
            self.response_verdicts,
            {
                "critical": False,
                "categories": [],
                "release": "allow",
                "reason_code": "judge_allowed",
                "explanation": "",
            },
        )

    @staticmethod
    def _next(queue: deque, default: Mapping[str, Any]) -> Mapping[str, Any]:
        value = queue.popleft() if queue else default
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)


@dataclass(frozen=True)
class JudgeResult:
    verdict: RequestVerdict | ResponseVerdict | None
    reason_code: str

    @property
    def failed(self) -> bool:
        return self.verdict is None


def controlled_reason(verdict: RequestVerdict | ResponseVerdict) -> str:
    """Persist a local decision code, never arbitrary model-generated reason text."""
    if isinstance(verdict, RequestVerdict):
        return "judge_allowed" if verdict.decision == "allow" else "judge_denied"
    if verdict.critical:
        return "judge_critical"
    return "judge_allowed" if verdict.release == "allow" else "judge_withheld"


class JudgeRunner:
    """Bound each separate judge invocation and convert failures to controlled codes."""

    def __init__(self, judge: Judge, timeout_seconds: float = 5.0) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.judge = judge
        self.timeout_seconds = timeout_seconds

    async def request(self, inputs: RequestInspection) -> JudgeResult:
        return await self._run(self.judge.inspect_request, inputs, RequestVerdict)

    async def response(self, inputs: ResponseInspection) -> JudgeResult:
        return await self._run(self.judge.inspect_response, inputs, ResponseVerdict)

    async def _run(self, method: Any, inputs: Any, schema: Any) -> JudgeResult:
        try:
            raw = await asyncio.wait_for(method(inputs), timeout=self.timeout_seconds)
            verdict = schema.model_validate(raw)
            return JudgeResult(verdict=verdict, reason_code=controlled_reason(verdict))
        except TimeoutError:
            return JudgeResult(verdict=None, reason_code="judge_timeout")
        except (ValidationError, TypeError, ValueError):
            return JudgeResult(verdict=None, reason_code="judge_invalid")
        except Exception:  # noqa: BLE001 - all provider failures must fail closed
            return JudgeResult(verdict=None, reason_code="judge_error")
