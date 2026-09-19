"""Run model-directed quote clarification with a resumable human conversation.
Live console mode demonstrates judgment; offline mode replays authored decisions.

AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import json
from pathlib import Path

from lg_report.platform.human_loop import HumanLoop, console_response
from lg_report.platform.model_config import configured_model
from lg_report.platform.sample_runtime import argument_parser, settings_for
from lg_report.report.recording import record_run
from lg_report.workflows.quote_request import build_workflow

from .test_case import ANSWERS, INITIAL_VALUES, make_simulated_model


# This app owns the adapter between CLI options, model selection, and HumanLoop.
# The workflow owns decision parsing, follow-up state, and cancellation; keeping
# that split lets live and offline runs exercise identical resumable transitions.
# The offline branch is intentionally constrained to the authored fixture, while
# live mode requires console input so a human can answer model-generated asks.
def main():
    """Help a requester clarify their quote scope and retain the conversation report.

    The CLI rejects combinations that would make the report misleading: a live
    model needs a console responder, and custom initial values need live mode.
    ``record_run`` then captures model decisions and human answers with the same
    output/accounting path used by the other samples.
    """
    parser = argument_parser(__file__, __doc__)
    parser.add_argument("--client", choices=["console", "static"], default="static")
    parser.add_argument(
        "--values", type=Path, help="JSON object with initial quote fields"
    )
    parser.add_argument(
        "--scenario", choices=["complete", "cancel"], default="complete"
    )
    args = parser.parse_args()
    if args.client == "console" and not args.live:
        parser.error(
            "Console conversation requires --live; offline decisions are scripted."
        )
    if args.live and args.client == "static":
        parser.error(
            "Use --client console with --live so a human answers model questions."
        )
    if args.values and not args.live:
        parser.error(
            "Custom requests require --live; offline decisions match only the fixture."
        )
    values = (
        json.loads(args.values.read_text(encoding="utf-8"))
        if args.values
        else dict(INITIAL_VALUES)
    )
    if not isinstance(values, dict):
        parser.error("--values must contain a JSON object")
    settings = settings_for(__file__, __doc__, args=args)
    model, provider, name = (
        configured_model()
        if settings.live
        else (make_simulated_model(), "demo", "scripted-chat")
    )
    if not settings.live:
        print(
            "OFFLINE SIMULATION: questions and decisions are scripted, not LLM reasoning."
        )
    answers = iter(ANSWERS)

    def scripted(payload):
        """Let the offline lesson continue through human follow-ups reproducibly.

        HumanLoop calls this with the interrupted node's question mapping.
        Return one authored answer string for use as the interrupt's resume value.

        The cancellation scenario deliberately terminates at the first prompt;
        otherwise ``ANSWERS`` is consumed in workflow order. Printing the
        payload keeps the simulated human boundary visible in console output.
        Exhausting the answers raises StopIteration; there is no fallback answer.
        """
        print(json.dumps(payload))
        return "/cancel" if args.scenario == "cancel" else next(answers)

    responder = console_response if args.client == "console" else scripted
    # HumanLoop construction runs no nodes. record_run invokes the adapter now;
    # it runs the graph, calls the responder outside each interrupted node, and
    # invokes the graph again with Command(resume=...). The interrupted ask node
    # restarts from its beginning; after its interrupt returns the answer, graph
    # edges lead to reassessment. This call finishes only when that loop ends.
    record_run(
        HumanLoop(build_workflow(model), responder),
        {"values": values},
        settings.output,
        settings.prices,
        provider=provider,
        model=name,
        title="Human follow-up · quote request",
        demo=not settings.live,
        include_output=settings.capture_content,
        overwrite=settings.overwrite,
    )
    print((settings.output / "report.html").resolve())


if __name__ == "__main__":
    main()
