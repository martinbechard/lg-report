"""Measure Agent elapsed execution and model throughput from recorded spans.

Outer Agent turn spans measure complete execution, including tools and retries.
Model-call spans separately retain throughput diagnostics for both API and CLI
transports. The legacy standalone scale remains available for older assessments;
comparison speed now uses elapsed Agent time exclusively.
AI attribution: Generated with AI assistance by Alex Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from reporting.schema import QAScore, Run


def _merged_intervals(intervals):
    """Combine overlapping spans so parallel work contributes elapsed time once."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def agent_elapsed_seconds(run: Run) -> float | None:
    """Measure complete Agent turn boundaries, including tools and orchestration.

    The conversation harness tags each top-level Agent invocation with its turn.
    Its span ends when the answer returns; gaps between invocations are user
    waits and are never charged to the Agent. Merge root intervals to avoid
    double counting parallel execution. Explicit user/judge intervals are
    removed even when nested within a turn. Every Agent model call must belong
    to a recorded turn root: missing boundaries cannot become a partial total.
    Token receipts are not needed to measure elapsed time.
    """
    if run.demo or run.status != 'ok':
        return None
    excluded_roles = {'user', 'qa'}
    by_id = {step.id: step for step in run.steps}
    roots = [step for step in run.steps if step.parent_id is None
             and step.kind in {'workflow', 'model'}
             and step.context.get('report_turn') is not None
             and step.context.get('model_role') not in excluded_roles]
    calls = [step for step in run.steps if step.kind == 'model'
             and step.context.get('model_role') not in excluded_roles]
    if not roots or not calls or any(step.status == 'incomplete' for step in roots):
        return None
    root_ids = {step.id for step in roots}
    for call in calls:
        ancestor = call
        while ancestor.parent_id is not None:
            ancestor = by_id[ancestor.parent_id]
        if ancestor.id not in root_ids or call.status == 'incomplete':
            return None
        # Malformed timestamps outside their parent would undercount work.
        if not ancestor.start_ns <= call.start_ns <= call.end_ns <= ancestor.end_ns:
            return None
    active = _merged_intervals((step.start_ns, step.end_ns) for step in roots)
    excluded = _merged_intervals((step.start_ns, step.end_ns) for step in run.steps
                                 if step.context.get('model_role') in excluded_roles)
    total = sum(end - start for start, end in active)
    total -= sum(max(0, min(end, other_end) - max(start, other_start))
                 for start, end in active for other_start, other_end in excluded)
    return total / 1_000_000_000


def assistant_performance(run: Run) -> dict:
    """Measure assistant request timing without user-agent or judge overhead.

    Effective output rate divides reported output (including reported reasoning)
    by summed model-call time. It includes transport, startup, and preprocessing;
    the shared boundary deliberately does not isolate decoding. Model time per turn is
    also a sum, not wall latency for parallel calls or tool-heavy workflows.
    Elapsed Agent time uses outer turn spans separately and needs no token
    receipt. Failed/static runs remain unavailable; missing usage affects only
    throughput, while missing boundaries prevent a complete elapsed total.
    A recovered model-format error in a successful workflow is still measured
    work: include its receipt and duration instead of hiding correction overhead.
    """
    calls = [step for step in run.steps if step.kind == "model"
             and step.context.get("model_role") not in {"user", "qa"}]
    result = {"tokens_per_second": None, "model_seconds_per_turn": None,
              "agent_elapsed_seconds": agent_elapsed_seconds(run)}
    if (run.demo or run.status != "ok" or not calls
            or any(step.status not in {"ok", "error"} for step in calls)):
        return result
    seconds = sum(step.duration_ms for step in calls) / 1000
    if seconds > 0 and all(step.usage is not None for step in calls):
        result["tokens_per_second"] = sum(step.usage.output_tokens for step in calls) / seconds
    turns = {step.context.get("report_turn") for step in calls}
    if None not in turns:
        result["model_seconds_per_turn"] = seconds / len(turns)
    return result


def speed_score(run: Run) -> QAScore:
    """Apply a monotonic, comparison-independent scale to successful live work.

    Equal components reward effective output throughput and low summed assistant
    model time per conversation turn. Both must be measured. Token throughput
    includes reported reasoning and overhead, so this is effective throughput,
    not pure decode rate. Tools, user agents, and the QA judge are excluded.
    """
    metrics = assistant_performance(run)
    rate, latency = metrics["tokens_per_second"], metrics["model_seconds_per_turn"]
    if rate is None or latency is None:
        return QAScore(score=None, reason="Speed unscored: complete successful live agent usage, timing, and turn attribution are required.")
    score = 50 * rate / (rate + 25) + 50 * 5 / (latency + 5)
    return QAScore(score=round(score, 1), reason=(
        f"Calculated from {rate:.2f} reported output tokens/s and {latency:.2f} agent model seconds/turn. "
        "Speed = 50 × rate/(rate + 25) + 50 × 5/(seconds per turn + 5). "

    ))
