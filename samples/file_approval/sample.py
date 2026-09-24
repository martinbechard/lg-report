"""Script model tool choices while exercising real reads, writes, and approvals.

SAMPLE declares discovery metadata alongside this scenario. Model factories
create fresh simulated models only when called; live mode uses the provider.

The fixture reacts to actual tool observations so a rejected write cannot be
mistaken for changed file content. This is authored behavior, not LLM reasoning.

AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import re

from langchain_core.messages import AIMessage, ToolMessage

from agent_runtime.harness.model_factory import client_prompts, model_responses
from agent_runtime.harness.simulated_model import MeteredDemoModel

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    "id": "file_approval",
    "name": "File approval",
    "description": "Review restricted writes proposed by the agent.",
    "interaction": "approval",
    "default_client": "console",
}

ADDITIONS = ["\nReviewed by the project team.\n", "Next step: request a quote.\n"]
REQUEST = "Append each of these sentences as a separate change: " + repr(ADDITIONS)


# Read the normal path top to bottom. Content placeholders are resolved from real
# reads, and human decisions resume approvals. A rejection can leave target.txt
# absent; the reactive adapter then proposes write_file again instead of edit_file.
# The default list has two additions, but the adapter also supports test overrides.
CONVERSATION = [
    {"role": "client", "content": REQUEST},
    {
        "role": "ai",
        "step": "read_source",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/source.txt"},
                "id": "file-call-{message_count}",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "content": "Expected: actual source text, including native pagination header.",
    },
    {
        "role": "ai",
        "step": "read_target",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/target.txt"},
                "id": "file-call-{message_count}",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "content": "Expected: current target text, or not-found on a fresh workspace.",
    },
    {
        "role": "ai",
        "step": "write_target",
        "content": "",
        "tool_calls": [
            {
                "name": "write_file",
                "args": {"file_path": "/target.txt", "content": "{source}\n{addition}"},
                "id": "file-call-{message_count}",
            }
        ],
    },
    {"role": "human", "interaction": "approval", "content": "approve"},
    {
        "role": "tool",
        "name": "write_file",
        "content": "Expected: real approval/write outcome; rejection does not create the target.",
    },
    {
        "role": "ai",
        "step": "reread_target",
        "content": "",
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/target.txt"},
                "id": "file-call-{message_count}",
            }
        ],
    },
    {
        "role": "tool",
        "name": "read_file",
        "content": "Expected: actual target after the first decision; never assume the write succeeded.",
    },
    {
        "role": "ai",
        "step": "edit_target",
        "content": "",
        "tool_calls": [
            {
                "name": "edit_file",
                "args": {
                    "file_path": "/target.txt",
                    "old_string": "{before}",
                    "new_string": "{before}\n{addition}",
                },
                "id": "file-call-{message_count}",
            }
        ],
    },
    {"role": "human", "interaction": "approval", "content": "approve"},
    {
        "role": "tool",
        "name": "edit_file",
        "content": "Expected: actual edit outcome, including human rejection.",
    },
    {
        "role": "ai",
        "step": "finish",
        "content": "Finished the requested proposals; see tool results for applied or rejected changes.",
    },
]
USER_PROMPTS = client_prompts(CONVERSATION)
# Explicit simulated human decision; live clients still ask the user.
APPROVAL_DECISION = next(
    entry["content"] for entry in CONVERSATION if entry["role"] == "human"
)


def scripted_step(name):
    """Copy one named AI template for resolution against actual tool observations."""
    return model_responses(
        [entry for entry in CONVERSATION if entry.get("step") == name]
    )[0]


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
        response = scripted_step("read_source")
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
                    response = scripted_step("read_target")
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
                        # The authored sample uses short, newline-terminated text.
                        # Pagination metadata is display-only and must not be saved.
                        response = scripted_step("write_target")
                        arguments = response.tool_calls[0]["args"]
                        arguments["content"] = arguments["content"].format(
                            source=read_body(source), addition=self.additions[writes]
                        )
                    else:
                        before = read_body(last)
                        response = scripted_step("edit_target")
                        arguments = response.tool_calls[0]["args"]
                        arguments["old_string"] = before
                        arguments["new_string"] = arguments["new_string"].format(
                            before=before, addition=self.additions[writes].rstrip("\n")
                        )
            else:
                response = scripted_step("reread_target")
        if writes >= len(self.additions):
            # Avoid claiming that rejected proposals were applied. The complete
            # transcript and decision audit show the exact outcomes in the report.
            response = scripted_step("finish")
        else:
            response.tool_calls[0]["id"] = response.tool_calls[0]["id"].format(
                message_count=len(messages)
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


def build_scripted_models(options):
    """Give the model factory fresh caller-specific scripted adapters for one run.

    The catalog supplies workflow options, not model instances. Actual tools
    still execute in the graph; these adapters author decisions and meter usage.
    """
    return {"workflow": make_simulated_model()}
