"""Define a LangGraph agent that runs requested scripts through the native execute tool.

ShellBackend supplies storage and command execution. The agent owns the prompt
and tool decisions; sample wiring supplies a prepared workspace and the model.
Commands run locally as the current user, without per-command approval in this
lesson. Only use this role with trusted requests and scripts.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents import create_agent

SYSTEM_PROMPT = """Run the shell script requested by the user using execute.
Commands run in the prepared working directory. Use relative paths in shell
commands; filesystem tools use virtual paths rooted in that same directory.
Report the actual command output and exit status. A nonzero exit or timeout is
not success. Do not invent script results or rerun a failed script unless asked.
"""


def build_agent(parameters: dict):
    """Construct the model/tool loop without executing a script or model call."""
    # Explicit middleware retains the supported execute schema and backend.
    # read_file is required by FilesystemMiddleware to retrieve evicted output;
    # other filesystem tools and implicit delegation are not part of this role.
    parameters = dict(parameters)
    backend = parameters.pop("backend")
    middleware = [*parameters.pop("middleware", []), FilesystemMiddleware(
        backend=backend, tools=["read_file", "execute"],
    )]
    return create_agent(
        **parameters,
        middleware=middleware,
        system_prompt=SYSTEM_PROMPT,
        name="shell_agent",
    )
