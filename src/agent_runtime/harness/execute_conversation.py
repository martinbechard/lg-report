"""Execute and record one prepared Python-client conversation.

The app module selects launch paths and supplies prepared settings and clients. SampleCatalog scopes
model selection while each workflow obtains its models from the factory. Students can
replace the reporting wrapper without rewriting the agent. Sharing this wrapper
also keeps all samples consistent about saved evidence. argument_parser owns
CLI options and validation; settings resolves configuration and accounting inputs.
Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from agent_runtime.harness.conversation import Conversation
from reporting.execute_runnable import execute_runnable

from .workflow_lifecycle import close_run, configure_context_audit, save_context_audit


def execute_conversation(
    *,
    catalog,
    sample_id,
    settings,
    options,
    client,
    env_file=None,
    show_context=False,
    public_trace=False,
):
    """Execute one prepared conversation through a Python client and recorder.

    The caller owns client creation, script configuration, and launch settings.
    Browser sessions use WebConversation for their own run/record/cleanup lifetime;
    this function neither starts an HTTP server nor parses command-line options.
    """
    import json
    from contextlib import ExitStack, nullcontext

    sample = catalog.get(sample_id)
    # Tracing adds callbacks and scopes to the same execution. Authenticate before
    # constructing models; the stack closes the SDK even if construction fails.
    with ExitStack() as resources:
        recorder = None
        scope = nullcontext(None)
        config = {}
        if sample.tracing == "langfuse":
            from .langfuse_runtime import LangfuseCapture, conversation_trace

            recorder = LangfuseCapture(
                settings=catalog.configuration(sample_id, env_file)
            )
            resources.callback(recorder.close)
            scope = conversation_trace(
                recorder.client,
                trace_name=sample.id.replace("_", "-"),
                simulated=not settings.live,
                public_trace=public_trace,
            )
            config = {
                "callbacks": [recorder.callback],
                "metadata": {
                    "report_description": "Execute the user turn, including any delegated work.",
                    "simulated": not settings.live,
                },
            }
        # The console owns its tracing scope above; prevent the catalog from
        # attaching a second recorder intended for independently streamed runs.
        graph, provider, model_id = catalog.create_run(
            sample_id, settings.live, options=options, env_file=env_file, tracing=False
        )
        try:
            configure_context_audit(
                graph,
                capture_content=settings.capture_content,
                show_context=show_context,
            )
            with scope as trace:
                # Construct the client-driven loop; no prompts or models run yet.
                conversation = Conversation(
                    graph, client, turn_scope=trace.turn_scope if trace else None
                )
                # Execution starts here: execute_runnable attaches capture, then calls
                # conversation.invoke({}, config). That call runs the entire loop:
                # receive a prompt, execute/resume the workflow, display its result,
                # and repeat until the client finishes or the workflow is cancelled.
                # execute_runnable saves the local report before returning (also on failure).
                result = execute_runnable(
                    conversation,
                    {},
                    settings.output,
                    settings.prices,
                    provider=provider,
                    model=model_id,
                    title=sample.name,
                    demo=not settings.live,
                    include_output=settings.capture_content,
                    overwrite=settings.overwrite,
                    config=config,
                )
                if trace:
                    trace.complete(result)
            if sample.interaction and getattr(client, "last_status", None) is not None:
                print(json.dumps({"status": client.last_status}))
            if recorder:
                # A URL identifies the trace; asynchronous ingestion is not verified here.
                recorder.client.flush()
                print(
                    f"Trace: {recorder.client.get_trace_url(trace_id=trace.trace_id)}"
                )
        finally:
            try:
                save_context_audit(graph, settings.output)
            finally:
                close_run(graph)
            if (settings.output / "report.html").exists():
                print((settings.output / "report.html").resolve())
