"""Select the shell agent for the local script-execution lesson.

The workflow prepares a private copy of the sample files and selects the
agent; the agent decides when to call DeepAgent's native execute tool. Building
this graph never pre-executes the script or fabricates its output.

AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from agent_runtime.agents.shell_agent import build_agent
from agent_runtime.backends.shell_backend import ShellBackend
from agent_runtime.harness.model_factory import build_model


def build_workflow(model=None, *, working_directory=None):
    """Prepare one session's workspace and retain it until the conversation closes.

    Direct tests may supply an existing directory. Normal clients get copied
    fixture files; the actual execute tool remains responsible for running them.
    Temporary directories prevent output collisions, not host-process access.
    """
    import shutil
    import stat
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import samples

    workspace = (
        TemporaryDirectory(prefix="lg-shell-script-")
        if working_directory is None
        else None
    )
    try:
        if workspace:
            working_directory = workspace.name
            for name in ("summarize.sh", "orders.csv"):
                shutil.copyfile(
                    Path(samples.__file__).parent / "shell_script" / name,
                    Path(working_directory) / name,
                )
            script = Path(working_directory) / "summarize.sh"
            script.chmod(script.stat().st_mode | stat.S_IXUSR)
        if model is None:
            model = build_model(caller="workflow")
        graph = build_agent(
            {"model": model, "backend": ShellBackend(working_directory)}
        )
    except BaseException:
        if workspace:
            workspace.cleanup()
        raise
    graph.workspace = workspace
    graph.output_file = Path(working_directory) / "summary.txt"
    return graph
