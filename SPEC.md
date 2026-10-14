# Deep Agents session trust middleware PoC

Status: implementation specification; application code has not been built.

## Goal

Build a small Python Deep Agents application with middleware that inspects every enabled tool request before execution and every tool response before it enters the main agent's context. Apply a deterministic policy first and an isolated LLM judge second.

A session begins with `trusted=True`. If either gate identifies critical returned data, persist `trusted=False` and restrict subsequent actions. Demonstrate the behavior using synthetic customer data and an outbound tool that writes only to a local outbox.

## Scope and terminology

`trusted` means that the session has not encountered critical data at the guarded tool boundary. It is a sensitivity marker, not evidence that a model, source, or instruction is trustworthy. The PoC tracks sensitive-data exposure; general prompt-injection detection is outside its guarantees.

Critical categories:

- `pii`: a person's identifying email address, phone number, or customer identifying record.
- `credential`: synthetic passwords, API keys, tokens, and private keys.
- `financial`: financial information linked to an identifiable person.
- `medical`: medical information linked to an identifiable person.
- `inspection_failed`: a returned payload whose sensitivity could not be determined safely.

An inspection failure conservatively taints the session. Keep it distinguishable from confirmed critical-data findings in audit events.

Initial scope: one agent, sequential tool calls, one process, text/JSON responses, in-memory checkpoints, and a command-line demo. Exclude subagents, arbitrary shell execution, real outbound sends, automatic declassification, production deployment, and scanning the agent's final answer. Final output protection can be a later extension.

## Security invariants

1. **Monotonic trust:** within a session, `True` may become `False`; `False` never becomes `True`. Public reads, retries, summaries, and later invocations do not clear taint.
2. **Policy precedence:** the judge may tighten a policy decision, but cannot override a denial or erase a critical finding.
3. **Execution boundary:** request checks finish before the handler executes. Denied requests invoke the handler zero times; approved requests invoke it exactly once.
4. **Response boundary:** inspect the complete supported response before releasing it to the main agent. Critical or uninspectable results are withheld and taint the session even though the main agent never sees the raw payload.
5. **Outbound restriction:** a tainted session cannot execute an external-write tool. This applies regardless of whether its proposed body appears harmless.
6. **Session isolation:** trust and audit state belong to the session/thread ID, not a global middleware-instance boolean. Reusing a session preserves taint; a new session starts clean.
7. **Fail closed:** request judge failure prevents execution. Response judge failure withholds the response and taints the session. Unsupported or oversized response types follow the response-failure path.
8. **Batch safety:** reject a batch with multiple tool calls before any of those tools execute. A provider setting or prompt requesting sequential execution alone is insufficient enforcement.
9. **Judge isolation:** the judge receives immutable inspection inputs, returns a validated structured verdict, and has no action tools or authority to update state.
10. **Audit privacy:** application audit logs contain decisions and metadata, not raw PII, credentials, tool arguments, or response bodies.

## Project structure

```text
pyproject.toml
uv.lock
README.md
src/trust_poc/
  agent.py          # agent assembly, checkpoint setup, session invocation
  middleware.py     # request/response gates and state updates
  policy.py         # tool registry and deterministic classification
  judge.py          # real model and fake judge adapters
  schemas.py        # state, findings, verdicts, audit events
  tools.py          # synthetic reads and local outbox
  demo.py           # reproducible scenario runner
tests/
fixtures/
```

Use `deepagents`, the chosen LangChain model integration, `pydantic`, and `pytest`. Use `uv` for dependencies and a committed lockfile. Pin the dependency set validated by the integration tests. Configure checking and demo commands in project configuration during implementation.

## State and contracts

Extend the supported agent state schema with:

```text
trusted: bool                         # default True
critical_categories: set[Category]    # default empty
security_events: list[SecurityEvent]  # default empty
```

Use a monotonic boolean reducer (`previous AND incoming`) for trust and a union reducer for categories. Initialize each fresh session explicitly. Verify reducer initialization with the pinned framework versions. Store all changes through framework-supported updates rather than mutating a request-state snapshot.

Policy findings contain `critical`, `categories`, and stable reason codes. Request policy additionally returns `allow | deny`; response policy returns `allow | withhold` for release.

Judge contracts:

```text
RequestVerdict:
  decision: allow | deny
  reason_code: string
  explanation: string

ResponseVerdict:
  critical: boolean
  categories: list[Category]
  release: allow | withhold
  reason_code: string
  explanation: string
```

Validate verdict consistency: a critical response is withheld; unknown categories and malformed verdicts fail closed. A noncritical response may still be withheld without confirmed critical categories. Uncertainty or an incomplete inspection uses `inspection_failed`.

Security events record session ID, tool-call ID, tool name, phase, gate outcomes, categories, trust before/after, and enforcement outcome. Use controlled reason codes in persistent events; free-text judge explanations can echo sensitive inputs and must not be logged raw. Keep the outbox independent of the audit log so tests can prove whether execution occurred.

## Tools and policy

Register three domain tools:

- `read_public_document()`: returns a public fixture; capability `local_read`.
- `read_customer_record(record_id)`: returns public, PII, credential, or ambiguous fixtures; capability `local_read`.
- `send_external_message(destination, body)`: records a send in a local outbox; capability `external_write`.

Capabilities and source labels are locally configured authority. Tool arguments and response text cannot redefine them. Unknown tools are denied. Remove unused default Deep Agents tools from the exposed tool set using APIs verified against the pinned version; explicitly allow any retained bookkeeping tools and include them in boundary tests.

Request policy evaluates the registered capability, argument validity, an explicit destination allowlist, recognizable sensitive content in outbound arguments, and current session taint. Deny disallowed destinations and any external write from a tainted session. Critical content in a proposed outbound body is denied even in a clean session.

Response policy scans every nested string in supported JSON and all supported text. Start with documented email/phone patterns, explicit synthetic credential formats, and trusted fixture/source metadata. Store detector findings as categories and reason codes without matching values. Record expected false positives and false negatives in README.md.

Set a concrete response size limit during implementation. Inspect the full bounded payload; oversized data is withheld and tainted rather than silently truncated. Exceptions exposed as tool results also pass through response inspection because error text can contain sensitive values.

## Judge

Use one separate model instance with structured output, configured by environment variables. A single model invocation per phase is sufficient; an autonomous judge agent is unnecessary for this PoC.

Request inputs: the original user task, exact proposed tool name/arguments, locally configured capability, current trust/categories, and policy findings. Evaluate task alignment, destination legitimacy, and potential disclosure.

Response inputs: tool/source metadata, the complete bounded response, policy findings, and the category definitions. Evaluate semantic sensitivity that simple patterns may miss, including identifying customer context and personal medical/financial facts.

Keep instructions in the judge's system prompt and inspected material in clearly delimited data fields. Give the judge a bounded timeout. Model-based judgments remain heuristic; hard policy findings remain authoritative.

Run the second gate on policy-permitted requests and fully inspectable responses. A deterministic request denial skips the judge and execution. A deterministic critical response still gets the second classification when feasible, but its finding is already binding. Inspection failures skip model evaluation when the payload cannot be processed safely.

The judge endpoint processes raw inspection inputs. Use only synthetic data for the demo and configure provider tracing so application artifacts do not capture raw inputs. Evaluating real data later requires a model endpoint authorized for that data.

## Middleware flow

Implement a custom `AgentMiddleware` using `awrap_tool_call` and attach it through `create_deep_agent(middleware=[...])`.

1. Resolve session identity and current security state.
2. Run request policy. On denial, emit an audit event and a denial `ToolMessage`; return without invoking the handler.
3. Invoke the request judge. Denial, timeout, or invalid output emits an event and denial message without execution.
4. Invoke the tool handler exactly once.
5. Normalize the returned message or supported command updates for inspection.
6. Run response policy, then response judge when inspection is feasible.
7. Combine findings: `critical = policy.critical OR judge.critical`; derive `trusted_after = trusted_before AND NOT critical`. Apply the conservative failure rule when inspection fails.
8. Release a safe result or replace it with a withholding message containing only category labels and reason codes.
9. Return a framework `Command` containing the corresponding `ToolMessage`, security state updates, and audit events.

Preserve the original tool-call ID in success, denial, and withholding messages. The agent should receive a clear reason for a blocked action without receiving withheld raw data.

If a handler returns a `Command`, inspect all content it would expose through messages or state. Preserve supported safe updates and merge security updates. Explicitly reject unsupported command shapes and fail closed; never drop arbitrary updates silently. Restrict the demo tools to simple responses, but test the adapter's rejection path.

Add a pre-execution batch guard at the model/tool scheduling boundary supported by the pinned version. When rejecting multiple calls, emit a matching denial message for every call and request sequential retry. Verify that no tool handlers execute. Monotonic reducers alone do not prevent a concurrent outbound write from reading stale trust.

## Implementation milestones

### 1. Establish the executable harness

Create dependency configuration, synthetic fixtures, the three tools, an independent outbox, and a fake judge interface. Assemble Deep Agents with only explicitly allowed tools and session checkpoints.

Complete when a public-read invocation runs, fresh/reused thread IDs are observable, and tests identify the exact exposed tool set and handler result shapes.

### 2. Implement policy and judge contracts

Add schemas, local tool metadata, deterministic request/response checks, structured judge validation, and timeout/error mapping.

Complete when tests cover each critical category, unknown tools, destination rules, policy precedence, malformed verdicts, and full nested-payload inspection.

### 3. Integrate request/response enforcement

Add middleware, checkpointed trust updates, safe audit events, withholding messages, and the enforced batch restriction.

Complete when integration tests prove handler execution counts, trust persistence across turns, fresh-session isolation, and zero execution for rejected batches.

### 4. Deliver the demo and live judge adapter

Add configured model evaluation and a scripted public-read → critical-read → blocked-send demonstration. Keep deterministic and live-model commands separate.

Complete when the deterministic suite passes, the demo reports policy/judge decisions and trust transitions, and README.md explains setup, commands, detector limitations, and any unverified live behavior.

## Acceptance scenarios

Use a fake judge for repeatable enforcement tests. Run live-model classification as a separate evaluation whose results are recorded without treating them as deterministic guarantees.

1. Public response passes both gates and leaves trust true.
2. Policy detects synthetic PII; result is withheld and trust becomes false.
3. Policy misses an ambiguous record; judge identifies critical data and trust becomes false.
4. Judge declares a policy-critical result safe; policy still wins.
5. A subsequent external write is denied and adds no outbox entry.
6. A later public read succeeds while trust stays false.
7. Request-policy denial invokes neither judge nor tool.
8. Request-judge denial invokes no tool.
9. Request-judge timeout or invalid verdict invokes no tool.
10. Response-judge failure withholds data and taints the session.
11. Oversized, unsupported, or partially inspectable results are withheld and taint the session.
12. A critical nested JSON field is detected and withheld.
13. Reusing a thread ID preserves false trust across agent invocations.
14. A new thread ID starts true and remains independent of other sessions.
15. A sensitive-read/send batch executes neither handler.
16. Response instructions asking the judge to ignore policy cannot erase deterministic findings.
17. Every returned tool message matches the originating call ID.
18. Audit artifacts contain no synthetic sensitive values, including values echoed in judge explanations.
19. Unsupported handler command updates fail closed rather than exposing uninspected data.
20. Every enabled tool is covered by the gate or explicitly justified bookkeeping policy.

## Completion and limits

The PoC is complete when all deterministic acceptance scenarios pass and its reproducible demo shows a blocked send with an empty outbox. Report live-model results separately, including missed classifications and failures.

The middleware governs intercepted calls; it is not host-wide containment. Response inspection cannot undo tool side effects. In-memory checkpoints do not survive process restart. The flag tracks tool-boundary sensitivity rather than all possible input channels, and final agent answers require a separate guard. Parallel execution, subagent propagation, durable checkpoints, richer classification levels, and human approval are follow-up work.

## API references

Verify API details against the pinned implementation versions:

- [Deep Agents customization](https://docs.langchain.com/oss/python/deepagents/customization): middleware, state schemas, tool configuration, checkpoints.
- [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom): interception hooks and state updates.
- [Tool interception API](https://reference.langchain.com/python/langchain/agents/middleware/types/AgentMiddleware/wrap_tool_call): request and handler contracts.
- [LangChain guardrails](https://docs.langchain.com/oss/python/langchain/guardrails): deterministic and model-based checks.
