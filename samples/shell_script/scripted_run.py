"""Script the model's command choice while consuming actual shell output.

The fixture never launches a process itself. DeepAgent executes the real tool,
and the fixture's final response quotes that observation, including failures.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain_core.messages import AIMessage, ToolMessage

from agent_runtime.harness.simulated_model import MeteredDemoModel

USER_PROMPTS = ["Run sh ./summarize.sh and report the order totals and exit status."]


class ShellFixture(MeteredDemoModel):
    """Author one execute call, then report its real observation."""

    def _generate(self, messages, *args, **kwargs):
        """Keep scripted model behavior separate from process execution."""
        last = messages[-1]
        if isinstance(last, ToolMessage):
            response = AIMessage(content=f"Shell tool returned:\n{last.content}")
        else:
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "args": {"command": "sh ./summarize.sh"},
                        "id": f"shell-call-{len(messages)}",
                    }
                ],
            )
        # Meter both model requests through the same fixture used by other
        # lessons. Usage is simulated; command output and side effects are real.
        self.responses = [response]
        self.i = 0
        return super()._generate(messages, *args, **kwargs)


def make_simulated_model():
    """Return a fresh fixture so separate sessions share no response state."""
    return ShellFixture(responses=[AIMessage(content="")])


def build_models(options):
    """Give the model factory fresh caller-specific scripted adapters for one run.

    The catalog supplies workflow options, not model instances. Actual tools
    still execute in the graph; these adapters author decisions and meter usage.
    """
    return {"workflow": make_simulated_model()}
