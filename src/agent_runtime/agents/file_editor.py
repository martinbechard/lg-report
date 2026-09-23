"""Own the file editor's instructions, tool registration, and model/tool loop.

The workflow supplies DeepAgents filesystem middleware, approval middleware, and
a checkpointer. LangChain runs the model/tool loop. FileAccessBackend restricts
the native tools to the configured documents.
Workflow middleware protects reviewed content from intervening changes.

AI attribution: Generated with AI assistance.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from langchain.agents import create_agent

SYSTEM_PROMPT = """You edit a UTF-8 document according to the human's request.
Use read_file to inspect /source.txt and /target.txt as needed. If the target exists,
preserve its current content unless the human requests replacement; otherwise
use the source as the starting document. Make each requested change separately,
with one write_file or edit_file call at a time, always on /target.txt.
Use write_file for new documents and edit_file for targeted replacements.
A missing target is normal: use the source to create it. Read results contain
pagination headers; these are display annotations, not document content.
After each write outcome, read the target again before proposing the next change.
A rejected tool call did not execute: do not retry or reintroduce that change.
Report only changes confirmed by tool results. Tool execution policy is enforced
by the harness; do not ask for permission in prose or invent approval results.
"""


def build_agent(parameters: dict):
    """Construct the editor without reading files or making any model calls."""
    # DeepAgents middleware supplies native file schemas. A plain LangChain
    # loop avoids an implicit delegate outside the workflow's approval policy.
    return create_agent(
        **parameters,
        system_prompt=SYSTEM_PROMPT,
        name="file_editor",
    )
