# Deep Agents session trust PoC

This project uses synthetic fixtures and a local outbox. No tool sends messages over the network.
The specification in [SPEC.md](SPEC.md) defines the security guarantees and implementation milestones.

Requires Python 3.12 or newer and `uv`.

```sh
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python -m trust_poc.agent
```

The harness runs without model credentials. It uses a scripted chat model, three domain tools,
and in-memory checkpoints selected by thread ID. The validated framework versions are pinned
in `pyproject.toml` and `uv.lock`. Runnable commands are also recorded in project configuration.

Deep Agents 0.7.25 supports replacing built-in middleware by its public `.name`. The harness
replaces filesystem, subagent, and summarization middleware with empty middleware, so the model
and graph expose exactly `read_public_document`, `read_customer_record`, and
`send_external_message`. No bookkeeping tools remain. Synthetic handlers return JSON dictionaries;
LangGraph's tool node wraps them as JSON-string `ToolMessage` results with the original call ID.

The assembled application always uses request and response gates; without an explicit judge it
uses a permissive fake judge for offline testing. Explicit fake verdicts demonstrate semantic
classification. In-memory checkpoints preserve taint across invocations of the same thread.
Live model evaluation remains separate from offline tests.

The model scheduling guard rejects multiple calls before scheduling any handler, returns a matching
denial for every call, and asks for a sequential retry. The tool adapter accepts only exact
content-format `StructuredTool` implementations, validates raw results before framework
stringification, and rejects unsupported tool implementations during assembly. Supported handler
Commands are local message-only updates with exactly one matching `ToolMessage`; all other state
updates and routing shapes fail closed. Supported safe message content, artifacts and metadata
are inspected together and preserved.

Audit events include decisions, categories, controlled reason codes, and stable SHA-256 references
for session and call IDs. They exclude tool arguments, payloads, model explanations and raw
identifiers. Tool messages retain their original call IDs. The local outbox independently records
actual synthetic send execution.

Deterministic inspection accepts text and JSON and bounds the complete serialized response to
16 KiB of UTF-8 with a maximum JSON depth of 32. It scans every nested string and JSON key,
including decoded JSON in tool message text. Unsupported objects, cycles, non-finite numbers,
and oversized responses fail closed with `inspection_failed`.

Email and phone patterns can flag public contact information or long numeric strings as PII.
Only documented `SYNTHETIC_API_KEY_`, `SYNTHETIC_PASSWORD_`, `SYNTHETIC_TOKEN_` formats and
private-key headers are credential patterns; arbitrary secrets and obfuscated values can be missed.
Locally configured fixture IDs identify PII, credential, financial and medical sources; returned
labels cannot change that authority. The ambiguous fixture deliberately relies on the judge.
Semantic medical/financial facts, indirect identity, encoded data and general prompt injection
are outside deterministic pattern guarantees. Model judgments remain heuristic.
