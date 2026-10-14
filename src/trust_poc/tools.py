"""Synthetic domain tools. The outbound operation only writes to a local outbox."""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures"


@dataclass
class LocalOutbox:
    """An execution record independent of security audit events."""

    entries: list[dict[str, str]] = field(default_factory=list)
    path: Path | None = None

    def append(self, destination: str, body: str) -> None:
        entry = {"destination": destination, "body": body}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
        self.entries.append(entry)


@dataclass
class DomainTools:
    tools: list[BaseTool]
    outbox: LocalOutbox
    execution_counts: Counter[str]


def build_tools(outbox: LocalOutbox | None = None) -> DomainTools:
    """Create separate synthetic handlers and counters for one application."""
    local_outbox = outbox if outbox is not None else LocalOutbox()
    counts: Counter[str] = Counter()
    public = json.loads((FIXTURE_ROOT / "public.json").read_text())
    records = json.loads((FIXTURE_ROOT / "customers.json").read_text())

    @tool
    def read_public_document() -> dict[str, Any]:
        """Read the public synthetic service desk document."""
        counts["read_public_document"] += 1
        return dict(public)

    @tool
    def read_customer_record(record_id: str) -> dict[str, Any]:
        """Read one synthetic record by its configured fixture ID."""
        counts["read_customer_record"] += 1
        if record_id not in records:
            return {"error": "unknown_record"}
        return dict(records[record_id])

    @tool
    def send_external_message(destination: str, body: str) -> dict[str, Any]:
        """Record a synthetic external message in the local outbox."""
        counts["send_external_message"] += 1
        local_outbox.append(destination, body)
        return {"status": "recorded_locally", "outbox_index": len(local_outbox.entries) - 1}

    return DomainTools(
        tools=[read_public_document, read_customer_record, send_external_message],
        outbox=local_outbox,
        execution_counts=counts,
    )
