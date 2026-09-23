"""Attach authored requests and optional interruption answers to a console client.

The caller selects script mode before invoking this configurator. It changes
user-side input on an existing client, independently of real or simulated model
responses. Each configuration owns fresh prompt and clarification cursors.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from .conversation import Request


def configure_sample_script(
    client,
    catalog,
    sample_id,
    *,
    prompts=None,
    script_answers=False,
    scenario="complete",
    decision=None,
):
    """Attach authored user input to an existing client, without executing a run.

    The caller decides whether to call this function; it reads no command-line
    arguments. prompts=None loads the sample's requests, while an explicit list
    replaces them, including an empty list. A fresh ScriptPrompter owns their
    sequence and stops at exhaustion.

    By default only prompts change: the client's existing interruption-answer
    behavior is preserved. script_answers=True also installs scripted approval
    or clarification answers for unattended runs. Those answers resume pauses
    within a turn and never advance the ordinary prompt sequence. Missing
    answers fail explicitly instead of falling back to terminal input.

    This mutates the supplied client and returns nothing. Model responses are
    configured separately through build_model; these are user-side inputs only.
    """
    from .script_prompter import ScriptPrompter

    sample = catalog.get(sample_id)

    def unexpected_answer(payload):
        """Keep unattended runs from silently reading stdin or granting consent."""
        raise ValueError("No scripted response for the pending interaction")

    answer = unexpected_answer
    if script_answers and sample.interaction:
        script = catalog.script(sample_id)
        # Each run owns its answer cursor. Reusing one across clients would make
        # a new conversation resume at the previous run's next clarification.
        answers = iter(getattr(script, "ANSWERS", []))

        def answer(payload):
            """Supply one scripted user decision, not an AI model response."""
            # Approval and clarification workflows use different cancellation
            # values. Preserve their contracts rather than interpreting them in
            # Conversation, which only forwards the response to the driver.
            if scenario == "cancel":
                return "cancel" if payload.get("kind") == "approval" else "/cancel"
            if sample.interaction == "approval":
                # An explicit CLI decision overrides the scenario's default.
                return decision or script.APPROVAL_DECISION
            # Exhaustion raises: an unexpected extra question must not silently
            # reuse a previous answer and make an incomplete script look valid.
            return next(answers)

    # Composition changes the source of ordinary requests, not the client:
    # ConsoleClient still presents results and handles interruption answers.
    # A missing scripted answer fails explicitly rather than reading stdin.
    requests = [
        Request(p) for p in (catalog.prompts(sample_id) if prompts is None else prompts)
    ]
    client.prompter = ScriptPrompter(requests)
    if script_answers:
        client.answer_callback = answer
