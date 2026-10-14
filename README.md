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

Run the deterministic demonstration from this source checkout:

```sh
uv run trust-poc-demo
# Optional: choose a separate artifact directory.
uv run trust-poc-demo --output-dir artifacts/example
```

It performs a public read, a withheld PII read, a blocked send, another public read in the tainted
session, and a public read in a fresh session. Each step prints policy/judge decisions, enforcement,
and trust transitions. Successful completion reports `blocked_send=True outbox_count=0`.
`artifacts/demo/audit.json` contains only security events, `summary.json` contains scenario outcomes,
and `outbox.jsonl` is an independent, empty send record. Repeating the demo overwrites these files.
An unexpected trust transition or executed send causes the demo to fail.
The current fixture loader expects `fixtures/` alongside `src/` in the source checkout; this project
is not packaged for running a standalone wheel outside that checkout.

The separate live evaluation requires an authorized model endpoint and uses only synthetic data:

```sh
# Copy .env.example and fill in the key and an explicit model; no default model is selected.
cp .env.example .env
# uv reads this file only when explicitly requested.
uv run --env-file .env trust-poc-evaluate
# Optional output destination:
uv run --env-file .env trust-poc-evaluate --output artifacts/custom-live-evaluation.json
```

`OPENAI_API_KEY` and `TRUST_POC_JUDGE_MODEL` are required. `TRUST_POC_JUDGE_TIMEOUT_SECONDS`
defaults to 5 and must be finite and positive. Use an available model supporting strict JSON Schema
outputs, as described in the [official OpenAI structured output documentation](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat).
The adapter creates a separate `ChatOpenAI` instance with no action tools and no retries. It sends
system instructions separately from a bounded JSON envelope of immutable inspection data, validates
structured request/response verdicts locally, and fails closed on refusals, malformed output,
timeouts, and provider errors. Complete judge input is capped at 64 KiB of UTF-8; oversized request
inspection inputs are denied. `JudgeRunner` applies the bounded timeout during middleware evaluation;
when assembling with a live judge, pass its `timeout_seconds` as `judge_timeout_seconds`.

The evaluation covers public, PII, credential, financial, medical, ambiguous, and instruction-bearing
responses plus legitimate and unaligned requests. The JSON report records expected and predicted
categories/decisions, controlled reason codes, misses, and failures. It excludes payloads, provider
errors, judge explanations, and free-text judge reasons. Exit status is 0 for matching evaluations,
1 for misses/failures, and 2 for missing or invalid configuration. Configuration failure records
`not_configured` without making a model call. Live model results are heuristic observations and
never replace deterministic acceptance tests.

Application graph calls and live judge calls explicitly disable LangSmith tracing even if enabled
in the environment. Do not add raw-input logging callbacks. The judge endpoint still receives raw
synthetic inspection data; later evaluation of real data requires an endpoint authorized for it.
Live classification has not been verified against a real provider in this implementation run;
model availability, endpoint compatibility, latency, and semantic classification quality remain
unverified. Adapter behavior is tested with an injected model stub, without network calls.

The configured `check` command runs Ruff linting, formatting verification, and pytest. All acceptance
scenarios in SPEC.md are covered by deterministic fake-judge tests:

- 1: `test_public_passes_two_gates_once_with_fresh_true`.
- 2, 4–6, 13–14: `test_policy_wins_persists_across_turns_and_new_session_is_clean`.
- 3, 18: `test_ambiguous_semantic_finding_and_private_judge_explanations` and
  `test_audit_identifiers_are_private_and_tool_message_preserves_original_id`.
- 7: `test_request_policy_denial_skips_judge_and_execution`.
- 8–9: `test_request_judge_denial_failure_never_executes` and
  `test_request_timeout_is_bounded_and_cancels_judge`.
- 10: `test_response_judge_failure_taints_and_withholds`.
- 11–12: `test_normalized_framework_tool_responses_are_fully_inspected`,
  `test_opaque_raw_tool_results_rejected_before_framework_stringification`,
  `test_async_raw_callback_is_validated_before_normalization`, and
  `test_all_nested_message_payloads_are_inspected`.
- 15: `test_batch_rejected_before_scheduling_and_sequential_retry_works`.
- 16: `test_returned_prompt_cannot_erase_policy_finding`.
- 17: originating call-ID assertions in enforcement and adapter tests.
- 19: `test_unsupported_command_is_rejected_through_actual_tool_node` and
  `test_uninspectable_shapes_fail_closed_and_skip_response_judge`.
- 20: `test_each_enabled_tool_is_gated` and exact exposed-tool assertions in `test_harness.py`.

The `trusted` flag tracks exposure at the guarded tool boundary. It does not establish that a model,
source, or instruction is trustworthy. The middleware cannot undo side effects of an approved tool,
guard arbitrary host actions, inspect the final agent answer, or track sensitive values arriving
through other input channels. Checkpoints last only for the process lifetime. Parallel execution,
subagent propagation, durable persistence, automatic declassification, richer sensitivity levels,
and human approval are outside this PoC. All outbound tool execution writes only to a local file.
