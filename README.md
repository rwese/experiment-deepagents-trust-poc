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

This first milestone establishes the executable harness. Enforcement, classification, and the
blocked-send demonstration are added in later milestones; the harness command currently proves
public reading and the restricted tool set. Live model evaluation remains separate from offline tests.
