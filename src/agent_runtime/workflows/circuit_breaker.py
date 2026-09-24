"""Stop repeated attempts to write a filename forbidden by the exercise backend.

This intentionally defective assignment reproduces the context-budget filename
mismatch. Real DeepAgents file tools reject each write; native LangChain call
limits end the agent before a fourth write executes. A stopped demonstration
is not successful implementation of the assigned file.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from tempfile import TemporaryDirectory

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)

from agent_runtime.harness.model_factory import build_model
from agent_runtime.workflows.exercise_backend import ExerciseBackend


def build_workflow(model=None, *, workspace_dir=None):
    """Bound a deliberately stubborn worker using native middleware.

    The write limit counts attempts, including backend rejections. It does not
    classify failures or infer progress. The model limit also bounds a model
    that stops requesting writes but keeps generating other activity. Counters
    belong to graph state, not summarizable conversation text, and reset each
    invocation. No provider calls occur during construction.
    """
    workspace = TemporaryDirectory(prefix="lg-circuit-breaker-") if workspace_dir is None else None
    try:
        directory = workspace.name if workspace else workspace_dir
        backend = ExerciseBackend(directory)
        if model is None:
            model = build_model(caller="worker")
        agent = create_agent(
            model=model,
            name="worker",
            system_prompt=(
                "You are the worker in an intentional circuit-breaker demonstration. "
                "The defective plan requires /slugify.py, but the backend only allows "
                "/plan.md, /slug.py, and /test_slug.py. Reproduce the failure by calling "
                "write_file for /slugify.py with the assigned content. After every "
                "rejection, request exactly the same write again. Do not change the "
                "path or content, and do not claim success. The application, not you, "
                "will stop repeated attempts through its call limit."
            ),
            middleware=[
                # DeepAgents requires read_file for its filesystem middleware.
                FilesystemMiddleware(backend=backend, tools=["read_file", "write_file"]),
                ToolCallLimitMiddleware(tool_name="write_file", run_limit=3, exit_behavior="end"),
                ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),
            ],
        )
        # Keep temporary storage alive for the same lifecycle used by the
        # context-budget sample; callers may instead retain an explicit folder.
        agent.workspace = workspace
        agent.workspace_dir = directory
        return agent
    except BaseException:
        if workspace:
            workspace.cleanup()
        raise
