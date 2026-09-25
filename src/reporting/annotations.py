"""Give captured operations human-readable purpose descriptions in reports.

Application-authored metadata takes priority; model calls fall back to the sample
description when available. Known sample and framework names provide further defaults. A generic description keeps
unrecognized operations visible. These labels describe the operation's role,
not private model reasoning or evidence that an intended action succeeded.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""


def describe(kind: str, name: str, metadata: dict, serialized: dict) -> str:
    """Help a report reader understand why a captured operation belongs in the run.

    Return one display description for the captured ``kind`` and runtime ``name``.
    ``metadata`` comes from callback configuration; ``serialized`` is the SDK
    definition, which can supply a tool description. Neither is executed here.

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
    # The catalog supplies the lesson's purpose as model-only fallback metadata.
    # It must not replace a tool's action or imply that every graph step performs
    # the whole sample. Missing descriptions retain the existing generic labels.
    if kind == "model" and metadata.get("report_sample_description"):
        return str(metadata["report_sample_description"])
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


def request_comment(step, *, summary: bool = False) -> str:
    """Describe a recorded model exchange without guessing its private reasoning.

    Prefer requested tool actions and their file/task targets. Otherwise show a
    bounded excerpt of the visible answer or user input. These are retrospective
    trace descriptions: a tool request does not establish successful execution.
    Missing capture stays explicit, and reasoning blocks are never used.
    """
    def excerpt(value):
        """Keep captured text compact and outside the tooltip's line delimiter."""
        text = " ".join(str(value).split()).replace(" | ", " / ")
        return text if len(text) <= 150 else text[:147] + "…"

    def visible_text(message):
        """Read only plain content and declared text blocks, never reasoning."""
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                block["text"] for block in content
                if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
                and isinstance(block.get("text"), str)
            )
        return ""

    if summary:
        return "Summarize conversation history for context compaction before the next model call."
    actions = []
    verbs = {"read_file": "reading", "write_file": "writing", "edit_file": "editing"}
    for message in step.response:
        for call in message.get("tool_calls", []):
            name = call.get("name", "unnamed tool")
            args = call.get("args", {})
            args = args if isinstance(args, dict) else {}
            if name in verbs and args.get("file_path"):
                action = f"{verbs[name]} {excerpt(args['file_path'])}"
            elif name == "task":
                action = f"delegation to {excerpt(args.get('subagent_type', 'subagent'))}"
                if args.get("description"):
                    action += f": {excerpt(args['description'])}"
            else:
                action = str(name)
            if action not in actions:
                actions.append(action)
    if actions:
        return "Request " + "; ".join(actions[:3]) + ("; additional tool calls" if len(actions) > 3 else "") + "."
    for message in reversed(step.response):
        text = visible_text(message)
        if text:
            return "Recorded answer: " + excerpt(text)
    for message in reversed(step.request):
        if message.get("role") in {"human", "user"}:
            text = visible_text(message)
            if text:
                return "Respond to: " + excerpt(text)
    return "Request purpose unavailable in the captured messages."
