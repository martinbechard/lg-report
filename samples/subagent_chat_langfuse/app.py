"""Trace a real parent/subagent handoff with the official Langfuse callback.

The graph and fixtures are shared with the local-report sample so this lesson
changes only the tracing backend. The task tool propagates callbacks into the
specialist; attaching another callback there would risk duplicate observations.
AI attribution: Generated with AI assistance.
"""

from lg_report.langfuse_runtime import launch
from samples.subagent_chat.app import USER_PROMPTS, create_graph


def main() -> None:
    """Run the delegation lesson with this directory's .env and a public trace."""
    launch(
        app_file=__file__,
        description=__doc__,
        create_graph=create_graph,
        prompts=USER_PROMPTS,
        trace_name="subagent-chat-langfuse",
    )


if __name__ == "__main__":
    main()
