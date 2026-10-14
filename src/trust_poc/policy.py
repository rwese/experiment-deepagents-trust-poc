"""Local authority and complete bounded deterministic payload inspection."""

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from trust_poc.schemas import (
    Capability,
    Category,
    RequestFinding,
    RequestVerdict,
    ResponseFinding,
    ResponseVerdict,
)

MAX_RESPONSE_BYTES = 16 * 1024
MAX_JSON_DEPTH = 32
ALLOWED_DESTINATIONS = frozenset({"review@example.test"})
RECORD_CATEGORIES: Mapping[str, tuple[Category, ...]] = MappingProxyType(
    {
        "public": (),
        "pii": ("pii",),
        "credential": ("credential",),
        "financial": ("financial",),
        "medical": ("medical",),
        "ambiguous": (),
    }
)


@dataclass(frozen=True)
class ToolMetadata:
    capability: Capability
    source_label: str


TOOL_REGISTRY: Mapping[str, ToolMetadata] = MappingProxyType(
    {
        "read_public_document": ToolMetadata("local_read", "public_fixture"),
        "read_customer_record": ToolMetadata("local_read", "synthetic_customer_fixture"),
        "send_external_message": ToolMetadata("external_write", "local_outbox_receipt"),
    }
)

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{7,}\d)(?!\w)")
CREDENTIAL = re.compile(
    r"SYNTHETIC_(?:API_KEY|PASSWORD|TOKEN)_[A-Z0-9_]+|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    re.IGNORECASE,
)


def source_categories(tool_name: str, arguments: Mapping[str, Any]) -> tuple[Category, ...]:
    if tool_name == "read_customer_record":
        record_id = arguments.get("record_id")
        return RECORD_CATEGORIES.get(record_id, ()) if isinstance(record_id, str) else ()
    return ()


def _scan(text: str) -> set[Category]:
    findings: set[Category] = set()
    if EMAIL.search(text) or PHONE.search(text):
        findings.add("pii")
    if CREDENTIAL.search(text):
        findings.add("credential")
    return findings


def request_policy(tool_name: str, arguments: Any, trusted: bool) -> RequestFinding:
    metadata = TOOL_REGISTRY.get(tool_name)
    if metadata is None:
        return RequestFinding(decision="deny", reason_codes=("unknown_tool",))
    if not isinstance(arguments, dict):
        return RequestFinding(decision="deny", reason_codes=("invalid_arguments",))
    expected = {
        "read_public_document": set(),
        "read_customer_record": {"record_id"},
        "send_external_message": {"destination", "body"},
    }[tool_name]
    if set(arguments) != expected or any(type(value) is not str for value in arguments.values()):
        return RequestFinding(decision="deny", reason_codes=("invalid_arguments",))
    if tool_name == "read_customer_record" and arguments["record_id"] not in RECORD_CATEGORIES:
        return RequestFinding(decision="deny", reason_codes=("unknown_record",))
    if metadata.capability == "external_write":
        reasons = []
        if not trusted:
            reasons.append("session_tainted")
        if arguments["destination"] not in ALLOWED_DESTINATIONS:
            reasons.append("destination_denied")
        categories = _scan(arguments["body"])
        if categories:
            reasons.append("outbound_sensitive")
        if reasons:
            return RequestFinding(
                decision="deny",
                critical=bool(categories),
                categories=tuple(sorted(categories)),
                reason_codes=tuple(reasons),
            )
    return RequestFinding(decision="allow", reason_codes=("policy_allowed",))


class InspectionFailure(ValueError):
    pass


def _walk(value: Any, strings: list[str], depth: int = 0, active: set[int] | None = None) -> None:
    if depth > MAX_JSON_DEPTH:
        raise InspectionFailure("depth")
    if active is None:
        active = set()
    if type(value) is str:
        strings.append(value)
    elif value is None or type(value) in (bool, int):
        return
    elif type(value) is float:
        if not math.isfinite(value):
            raise InspectionFailure("nonfinite")
    elif type(value) in (dict, list):
        if id(value) in active:
            raise InspectionFailure("cycle")
        active.add(id(value))
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise InspectionFailure("key")
                strings.append(key)
                _walk(item, strings, depth + 1, active)
        else:
            for item in value:
                _walk(item, strings, depth + 1, active)
        active.remove(id(value))
    else:
        raise InspectionFailure("unsupported")


def response_policy(
    payload: Any,
    *,
    tool_name: str = "read_public_document",
    arguments: Mapping[str, Any] | None = None,
) -> ResponseFinding:
    strings: list[str] = []
    try:
        _walk(payload, strings)
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(serialized.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise InspectionFailure("oversized")
        # ToolNode serializes dictionaries into text. Inspect JSON keys/values after
        # decoding, in addition to scanning the exact returned string.
        if type(payload) is str:
            try:
                nested = json.loads(payload)
            except (ValueError, RecursionError):
                nested = None
            if nested is not None:
                _walk(nested, strings)
    except (InspectionFailure, ValueError, TypeError, UnicodeError, RecursionError):
        return ResponseFinding(
            release="withhold",
            critical=True,
            categories=("inspection_failed",),
            reason_codes=("response_inspection_failed",),
            inspectable=False,
        )
    categories = set(source_categories(tool_name, arguments or {}))
    reasons = ["source_critical"] if categories else []
    for text in strings:
        detected = _scan(text)
        if detected:
            categories.update(detected)
            reasons.append("pattern_critical")
    return ResponseFinding(
        release="withhold" if categories else "allow",
        critical=bool(categories),
        categories=tuple(sorted(categories)),
        reason_codes=tuple(sorted(set(reasons))) if reasons else ("policy_allowed",),
        inspectable=True,
        payload_json=serialized,
    )


def combine_response(policy: ResponseFinding, verdict: ResponseVerdict) -> ResponseFinding:
    """The model can only tighten policy sensitivity and release decisions."""
    categories = set(policy.categories) | set(verdict.categories)
    critical = policy.critical or verdict.critical
    return ResponseFinding(
        release="withhold"
        if policy.release == "withhold" or verdict.release == "withhold"
        else "allow",
        critical=critical,
        categories=tuple(sorted(categories)),
        reason_codes=policy.reason_codes,
        inspectable=policy.inspectable,
        payload_json=policy.payload_json,
    )


def combine_request(policy: RequestFinding, verdict: RequestVerdict) -> RequestFinding:
    """A judge denial is binding; policy denials and findings remain binding."""
    return RequestFinding(
        decision="deny" if policy.decision == "deny" or verdict.decision == "deny" else "allow",
        critical=policy.critical,
        categories=policy.categories,
        reason_codes=policy.reason_codes,
    )
