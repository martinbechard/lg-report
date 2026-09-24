"""Script the model's command choice while consuming actual shell output.

SAMPLE declares discovery metadata alongside this scenario. Model factories
create fresh simulated models only when called; live mode uses the provider.

The fixture never launches a process itself. DeepAgent executes the real tool,
and the fixture's final response quotes that observation, including failures.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.messages import AIMessage, ToolMessage

from agent_runtime.harness.model_factory import client_prompts, model_responses
from agent_runtime.harness.simulated_model import MeteredDemoModel

# Discovery reads this metadata without constructing a model.
SAMPLE = {
    "id": "shell_script",
    "name": "Shell script",
    "description": "Run a local script through DeepAgent’s native execute tool.",
}

# The final AI step is a template filled from the actual tool result. It must
# report failures too; a canned successful result would hide shell errors.
CONVERSATION = [
    {
        "role": "client",
        "content": "Run sh ./summarize.sh and report the order totals and exit status.",
    },
    {
        "role": "ai",
        "content": "",
        "tool_calls": [
            {
                "name": "execute",
                "args": {"command": "sh ./summarize.sh"},
                "id": "shell-call-{message_count}",
            }
        ],
    },
    {
        "role": "tool",
        "name": "execute",
        "content": "Expected: actual stdout, stderr, and exit status; success is not assumed.",
    },
    {"role": "ai", "content": "Shell tool returned:\n{tool_result}"},
]
USER_PROMPTS = client_prompts(CONVERSATION)


class ShellFixture(MeteredDemoModel):
    """Author one execute call, then report its real observation."""

    def _generate(self, messages, *args, **kwargs):
        """Keep scripted model behavior separate from process execution."""
        last = messages[-1]
        steps = model_responses(CONVERSATION)
        if isinstance(last, ToolMessage):
            response = steps[1]
            response.content = response.content.format(tool_result=last.content)
        else:
            response = steps[0]
            response.tool_calls[0]["id"] = response.tool_calls[0]["id"].format(
                message_count=len(messages)
            )
        # Meter both model requests through the same fixture used by other
        # lessons. Usage is simulated; command output and side effects are real.
        self.responses = [response]
        self.i = 0
        return super()._generate(messages, *args, **kwargs)


def make_simulated_model():
    """Return a fresh fixture so separate sessions share no response state."""
    return ShellFixture(responses=[AIMessage(content="")])


def build_scripted_models(options):
    """Give the model factory fresh caller-specific scripted adapters for one run.

    The catalog supplies workflow options, not model instances. Actual tools
    still execute in the graph; these adapters author decisions and meter usage.
    """
    return {"workflow": make_simulated_model()}
