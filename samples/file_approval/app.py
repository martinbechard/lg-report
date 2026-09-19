"""Run file editing with automatic approval or a human decision for every write.
The console is the default; static decisions are explicitly simulated approvals.

AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from lg_report.platform.human_loop import HumanLoop, console_response
from lg_report.platform.sample_runtime import argument_parser, settings_for
from lg_report.report.recording import record_run
from lg_report.workflows.file_approval import build_workflow

from .test_case import ADDITIONS


# This entry point is deliberately deterministic: the workflow has no model,
# and always-ask pauses in HumanLoop before each write; autoapprove skips asking.
# The static client demonstrates approval decisions without touching a provider;
# console mode is the integration path for an actual reviewer.
def main():
    """Demonstrate how per-write decisions control the prepared file changes.

    Read CLI paths and approval policy, run both additions through the workflow,
    and save a report. Static answers are simulated decisions but approved writes
    still modify the real target. Autoapprove bypasses the responder entirely.
    """
    parser = argument_parser(__file__, __doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--mode", choices=["autoapprove", "always-ask"], default="always-ask"
    )
    parser.add_argument("--client", choices=["console", "static"], default="console")
    parser.add_argument(
        "--decision", choices=["approve", "reject", "cancel"], default="approve"
    )
    args = parser.parse_args()
    if args.live:
        parser.error("This deterministic workflow does not use an LLM")
    settings = settings_for(__file__, __doc__, args=args)
    if args.client == "static":
        print("Simulated human decisions:", args.decision)
    # HumanLoop passes the proposed-change payload to the responder on a pause.
    # This lambda accepts that payload but deliberately ignores its contents:
    # every simulated decision uses the same CLI choice, not a review judgment.
    responder = (
        console_response if args.client == "console" else lambda payload: args.decision
    )
    # Constructing HumanLoop only wires the graph and responder. record_run
    # invokes it now; it handles pauses and re-invocations until the graph ends,
    # then the recorder finishes the report before this function prints its path.
    record_run(
        HumanLoop(build_workflow(), responder),
        {
            "source": args.source,
            "target": args.target,
            "mode": args.mode,
            "additions": ADDITIONS,
        },
        settings.output,
        settings.prices,
        provider="none",
        model="none",
        title="Human approval · file modifications",
        demo=args.client == "static",
        include_output=settings.capture_content,
        overwrite=settings.overwrite,
    )
    print((settings.output / "report.html").resolve())


if __name__ == "__main__":
    main()
