"""Compute repeatable assistant speed measurements and a fixed standalone score.

The common boundary is the application's complete model invocation, available
for both API and CLI transports. Internal backend timings are never substituted into only one model's score. Anchors are application
calibration choices, not industry benchmarks, SLAs, or competitor rankings.
AI attribution: Generated with AI assistance by Alex Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from reporting.schema import QAScore, Run


def assistant_performance(run: Run) -> dict:
    """Measure assistant request timing without user-agent or judge overhead.

    Effective output rate divides reported output (including reported reasoning)
    by summed model-call time. It includes transport, startup, and preprocessing;
    the shared boundary deliberately does not isolate decoding. Model time per turn is
    also a sum, not wall latency for parallel calls or tool-heavy workflows.
    Failed/static runs and absent usage or turn attribution remain unavailable.
    """
    calls = [step for step in run.steps if step.kind == "model"
             and step.context.get("model_role") not in {"user", "qa"}]
    result = {"tokens_per_second": None, "model_seconds_per_turn": None}
    if run.demo or run.status != "ok" or not calls or any(step.status != "ok" for step in calls):
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
