"""Application-owned descriptions for the sample operations and framework spans."""


def describe(kind: str, name: str, metadata: dict, serialized: dict) -> str:
    explicit = metadata.get("report_description") or serialized.get("description")
    if explicit:
        return str(explicit)
    samples = {
        "chat-agent": "Coordinate a direct chat response: prepare messages and invoke the model.",
        "reference-chat-agent": "Coordinate reference-assisted chat: request a local reference, execute the lookup, and answer using its result.",
        "PatchToolCallsMiddleware.before_agent": "Check message history and repair missing tool responses before the agent starts.",
        "model": "Prepare the current conversation, run model middleware, and invoke the chat model.",
        "tools": "Execute tool calls requested by the preceding model response and return their observations.",
    }
    if name in samples:
        return samples[name]
    if kind == "model":
        return "Generate the next assistant message or tool request from the current conversation."
    if kind == "tool":
        return f"Execute the {name} tool and return its result to the agent."
    if kind == "retriever":
        return f"Retrieve documents using {name} for the current query."
    return f"Execute the {name} workflow operation and pass its result to the next graph step."
