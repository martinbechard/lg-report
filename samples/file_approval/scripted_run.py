"""Script model tool choices while exercising real reads, writes, and approvals.

The fixture reacts to actual tool observations so a rejected write cannot be
mistaken for changed file content. This is authored behavior, not LLM reasoning.

AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import re

from langchain_core.messages import AIMessage, ToolMessage

from agent_runtime.harness.simulated_model import MeteredDemoModel

ADDITIONS = ["\nReviewed by the project team.\n", "Next step: request a quote.\n"]
REQUEST = "Append each of these sentences as a separate change: " + repr(ADDITIONS)


def read_body(message):
    """Decode the native read header for this fixture's short text documents.

    Fail visibly if the library's display protocol changes; never treat an
    error or truncated result as replacement document content.
    """
    match = re.fullmatch(
        r"@@ lines 1-(\d+) of (\d+) @@\n(.*)", message.content, re.DOTALL
    )
    if message.status != "success" or not match or match[1] != match[2]:
        raise ValueError("Fixture requires a complete native text read")
    return match[3]


class FileEditingFixture(MeteredDemoModel):
    """Author the demo's read/write choices using observations from real tools."""

    additions: list[str]

    def _generate(self, messages, *args, **kwargs):
        """Choose the next scripted response; let the shared meter record it."""
        # Count proposals, not successful writes: a rejected change is consumed
        # by this scenario and is not automatically retried in the next turn.
        writes = sum(
            call["name"] in {"write_file", "edit_file"}
            for message in messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        )
        last = messages[-1]
        name, arguments = "read_file", {"file_path": "/source.txt"}
        if isinstance(last, ToolMessage):
            if last.name == "read_file":
                # Native results contain a pagination header rather than our old
                # custom JSON envelope. Match each result to its real tool call.
                calls = {
                    call["id"]: call
                    for message in messages
                    if isinstance(message, AIMessage)
                    for call in message.tool_calls
                }
                path = calls[last.tool_call_id]["args"]["file_path"]
                if path == "/source.txt":
                    arguments = {"file_path": "/target.txt"}
                elif writes < len(self.additions):
                    missing = last.status == "error" and "not found" in last.content
                    if last.status == "error" and not missing:
                        raise RuntimeError(last.content)
                    if missing:
                        source = next(
                            item
                            for item in messages
                            if isinstance(item, ToolMessage)
                            and item.name == "read_file"
                            and calls[item.tool_call_id]["args"]["file_path"]
                            == "/source.txt"
                        )
                        name = "write_file"
                        # The authored sample uses short, newline-terminated text.
                        # Pagination metadata is display-only and must not be saved.
                        arguments = {
                            "file_path": "/target.txt",
                            "content": read_body(source)
                            + "\n"
                            + self.additions[writes],
                        }
                    else:
                        before = read_body(last)
                        name = "edit_file"
                        arguments = {
                            "file_path": "/target.txt",
                            "old_string": before,
                            "new_string": before
                            + "\n"
                            + self.additions[writes].rstrip("\n"),
                        }
            else:
                arguments = {"file_path": "/target.txt"}
        if writes >= len(self.additions):
            # Avoid claiming that rejected proposals were applied. The complete
            # transcript and decision audit show the exact outcomes in the report.
            response = AIMessage(
                content="Finished the requested proposals; see tool results for applied or rejected changes."
            )
        else:
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": name,
                        "args": arguments,
                        "id": f"file-call-{len(messages)}",
                    }
                ],
            )
        # Reuse metering/callback behavior; only model choice is simulated. The
        # fixture never opens a file or executes a tool behind the graph's back.
        self.responses = [response]
        self.i = 0
        return super()._generate(messages, *args, **kwargs)


def make_simulated_model(additions=None):
    """Create an isolated script and meter for one editing conversation."""
    # A fresh instance prevents response position and usage leaking across runs.
    return FileEditingFixture(
        responses=[AIMessage(content="")],
        additions=ADDITIONS if additions is None else additions,
    )


def build_models(options):
    """Give the model factory fresh caller-specific scripted adapters for one run.

    The catalog supplies workflow options, not model instances. Actual tools
    still execute in the graph; these adapters author decisions and meter usage.
    """
    return {"workflow": make_simulated_model()}


USER_PROMPTS = [REQUEST]

# Explicit simulated human decision; live clients still ask the user.
APPROVAL_DECISION = "approve"
