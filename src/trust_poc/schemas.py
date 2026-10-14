"""Immutable inspection contracts and session-owned security state."""

import operator
from typing import Annotated, Literal, Self

from deepagents.graph import DeepAgentState
from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["pii", "credential", "financial", "medical", "inspection_failed"]
Capability = Literal["local_read", "external_write"]


def monotonic_trust(previous: bool | None, incoming: bool | None) -> bool:
    """None is the empty-channel identity; false can never become true."""
    return (True if previous is None else previous) and (True if incoming is None else incoming)


def union_categories(previous: set[Category], incoming: set[Category]) -> set[Category]:
    return previous | incoming


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RequestFinding(FrozenContract):
    decision: Literal["allow", "deny"]
    critical: bool = False
    categories: tuple[Category, ...] = ()
    reason_codes: tuple[str, ...] = ()


class ResponseFinding(FrozenContract):
    release: Literal["allow", "withhold"]
    critical: bool = False
    categories: tuple[Category, ...] = ()
    reason_codes: tuple[str, ...] = ()
    inspectable: bool = True
    payload_json: str | None = None


class RequestVerdict(FrozenContract):
    decision: Literal["allow", "deny"]
    reason_code: str = Field(min_length=1, max_length=128)
    explanation: str = Field(max_length=4096)


class ResponseVerdict(FrozenContract):
    critical: bool
    categories: list[Category]
    release: Literal["allow", "withhold"]
    reason_code: str = Field(min_length=1, max_length=128)
    explanation: str = Field(max_length=4096)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.critical and (not self.categories or self.release != "withhold"):
            raise ValueError("critical verdicts require categories and withholding")
        if not self.critical and self.categories:
            raise ValueError("noncritical verdicts cannot contain critical categories")
        return self


class RequestInspection(FrozenContract):
    task: str
    tool_name: str
    arguments_json: str
    capability: Capability
    trusted: bool
    categories: tuple[Category, ...]
    policy: RequestFinding


class ResponseInspection(FrozenContract):
    tool_name: str
    source_label: str
    source_categories: tuple[Category, ...]
    payload_json: str
    policy: ResponseFinding
    category_definitions: tuple[str, ...] = (
        "pii: identifying email, phone, or customer record",
        "credential: password, API key, token, or private key",
        "financial: financial information linked to an identifiable person",
        "medical: medical information linked to an identifiable person",
        "inspection_failed: sensitivity could not be determined safely",
    )


class SecurityEvent(FrozenContract):
    session_id: str
    tool_call_id: str
    tool_name: str
    phase: Literal["request", "response", "batch"]
    policy_outcome: Literal["allow", "deny", "withhold", "not_run"]
    judge_outcome: Literal["allow", "deny", "withhold", "failed", "not_run"]
    reason_codes: tuple[str, ...]
    categories: tuple[Category, ...]
    trusted_before: bool
    trusted_after: bool
    enforcement: Literal["executed", "denied", "released", "withheld", "batch_rejected"]


class SecurityState(DeepAgentState):
    # A union avoids bool() seeding the framework reducer with False.
    trusted: Annotated[bool | None, monotonic_trust]
    critical_categories: Annotated[set[Category], union_categories]
    security_events: Annotated[list[SecurityEvent], operator.add]
