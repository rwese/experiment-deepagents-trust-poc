"""Isolated asynchronous judge interface for the executable harness."""

from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Protocol


class Judge(Protocol):
    async def inspect_request(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def inspect_response(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]: ...


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

    async def inspect_request(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        self.request_calls += 1
        return self._next(
            self.request_verdicts,
            {"decision": "allow", "reason_code": "judge_allowed", "explanation": ""},
        )

    async def inspect_response(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
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
