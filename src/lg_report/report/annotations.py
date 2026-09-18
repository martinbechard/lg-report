"""Give captured operations human-readable purpose descriptions in reports.

Application-authored metadata takes priority; known sample and framework names
have descriptions when explicit metadata is absent. A generic description keeps
unrecognized operations visible. These labels describe the operation's role,
not private model reasoning or evidence that an intended action succeeded.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""


def describe(kind: str, name: str, metadata: dict, serialized: dict) -> str:
    """Explain an operation using application intent before generic framework roles.

    report_description is the application's annotation; serialized description
    commonly comes from a tool definition. Generic descriptions explain the
    operation's role only: they do not claim to reveal the model's reasoning.
    This function returns display text and never modifies a model prompt.
    """
    explicit = metadata.get("report_description") or serialized.get("description")
    # An application annotation (or tool description when none is supplied)
    # describes intent more accurately than our generic role labels. Empty text
    # carries no useful intent, so continue to known names and role defaults.
    if explicit:
        return str(explicit)
    samples = {
        "chat_agent": "Coordinate a direct chat response: prepare messages and invoke the model.",
        "chat-agent": "Coordinate a direct chat response: prepare messages and invoke the model.",
        "reference-chat-agent": "Coordinate reference-assisted chat: request a local reference, execute the lookup, and answer using its result.",
        "PatchToolCallsMiddleware.before_agent": "Check message history and repair missing tool responses before the agent starts.",
        "model": "Prepare the current conversation, run model middleware, and invoke the chat model.",
        "tools": "Execute tool calls requested by the preceding model response and return their observations.",
    }
    # Known sample/framework names let us explain their particular job; unknown
    # operations must use their captured kind rather than a guessed purpose.
    if name in samples:
        return samples[name]
    # A model span produces messages or tool requests; it does not execute the
    # tools itself. Keep that boundary distinct from the tool branch below.
    if kind == "model":
        return "Generate the next assistant message or tool request from the current conversation."
    # A tool span is the actual action requested by a model, not another LLM call.
    if kind == "tool":
        return f"Execute the {name} tool and return its result to the agent."
    # Retrieval specifically returns documents; remaining kinds represent
    # workflow orchestration and receive the generic workflow description.
    if kind == "retriever":
        return f"Retrieve documents using {name} for the current query."
    return f"Execute the {name} workflow operation and pass its result to the next graph step."
