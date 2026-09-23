"""Present conversations using terminal input or an authored prompt sequence.

Files are read only when the user attaches them; nothing grants the agent access
to the surrounding filesystem. No web server or extra UI dependencies are needed.
When given a catalog, this client also handles sample-navigation commands;
ConsoleApplication performs the requested restart or switch after the conversation
ends. Interruption answers are never interpreted as navigation commands.
Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from pathlib import Path

from .conversation import Attachment, Request
from .script_prompter import ScriptPrompter


class ConsoleClient:
    """Use /attach PATH before a prompt, /send for files alone, and /quit to end.

    Only UTF-8 text is supported in this first sample. read/write are injectable
    to test console behavior without a terminal or provider call.
    """

    def __init__(
        self,
        read=input,
        write=print,
        *,
        catalog=None,
        sample_id=None,
        prompter: ScriptPrompter | None = None,
        answer=None,
    ):
        """Prepare an interactive user client, with replaceable I/O for tests.

        read receives the prompt label and returns one line. write displays one
        string. Attachments stay queued until a request is actually submitted.
        An optional prompter supplies requests instead of reading the terminal;
        exhaustion ends a standalone conversation. With a catalog, exhaustion
        returns to terminal input so the user can restart or select a sample.
        catalog and sample_id enable navigation; next_sample records the choice
        for ConsoleApplication to launch after the current conversation closes.
        answer optionally supplies user responses to workflow interruptions. It
        receives the payload unchanged and does not advance the prompt sequence.
        Without that callback, interruption answers are read from the terminal.
        """
        self.catalog = catalog
        self.sample_id = sample_id
        self.next_sample = None
        self.structured = bool(catalog.get(sample_id).interaction) if catalog else False
        self.read = read
        self.write = write
        self.files = []
        self.prompter = prompter
        self.answer_callback = answer
        self.last_status = None

    def receive(self) -> Request | None:
        """Take the next authored request, or collect terminal commands and input.

        Returns a ``Request`` for a prompt (possibly with queued attachments),
        or ``None`` for explicit quit/EOF/Ctrl-C. Blank input and ``/send`` with
        no files do not create turns. Attachment I/O errors are reported through
        ``write`` and leave previously queued files available; non-UTF-8 files
        are rejected because the conversation boundary accepts text only.
        """
        if self.prompter is not None:
            request = self.prompter.next()
            if request is not None or self.catalog is None:
                return request
            # Interactive sample selection stays available after authored prompts;
            # static runs omit the catalog and finish when their prompts end.
            self.prompter = None
        while True:
            try:
                line = self.read("User: ")
            except (EOFError, KeyboardInterrupt):
                # EOF/Ctrl-C at the prompt is a normal user exit. Returning lets
                # the recorder close spans and generate the session report.
                return None
            command = line.strip()
            if self.catalog is not None:
                # Navigation is handled only between turns. answer() reads directly
                # so slash text in an interruption response stays workflow input.
                if command == "/samples":
                    for sample in self.catalog.samples.values():
                        self.write(f"{sample.id}: {sample.name} — {sample.description}")
                    continue
                if command == "/help":
                    self.write(
                        "/samples · /sample ID · /new · /attach PATH · /send · /quit"
                    )
                    continue
                if command == "/new" or command.startswith("/sample "):
                    selected = (
                        self.sample_id
                        if command == "/new"
                        else command.split(maxsplit=1)[1]
                    )
                    try:
                        self.catalog.get(selected)
                    except ValueError as exc:
                        self.write(str(exc))
                        continue
                    # End this conversation first so recording and graph cleanup
                    # finish before ConsoleApplication starts the next sample.
                    self.next_sample = selected
                    return None
                if self.structured and command != "/quit":
                    self.write(
                        "This structured run is complete. Use /new, /sample ID, or /quit."
                    )
                    continue
            # Explicit exit leaves queued attachments unsent; attaching a file alone
            # is not permission to submit its contents to the model.
            if command == "/quit":
                return None
            # Attachment commands queue file contents without creating a turn,
            # so several files can accompany the following user prompt.
            if command.startswith("/attach "):
                path = Path(command.removeprefix("/attach ").strip()).expanduser()
                try:
                    content = path.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    # A bad path or non-text file must not discard queued files
                    # or call the model with a pretend attachment.
                    self.write(f"Cannot attach UTF-8 text file ({type(exc).__name__}).")
                    continue
                self.files.append(Attachment(path.name, content))
                self.write(f"Attached {path.name}; enter a prompt or /send.")
                continue
            if command.startswith("/") and command != "/send":
                # Keep mistyped commands out of the model context.
                self.write("Commands: /attach PATH, /send, /quit")
                continue
            if not command or (command == "/send" and not self.files):
                # Blank input and an empty attachment-only send create no turn.
                continue
            request = Request("" if command == "/send" else line, tuple(self.files))
            self.files.clear()
            return request

    def respond(self, result: dict) -> None:
        """Display the completed turn; pending interactions use answer()."""
        # The client event pump calls this only after the driver has completed
        # all pauses for this turn. It does not ask the model to grant approval.
        # Retain only the completion status needed by the CLI summary. Full
        # result capture belongs to recording or the test-only MockClient.
        self.last_status = result.get("status")
        if result.get("messages"):
            self.write(f"Assistant: {result['messages'][-1].content}")

    def answer(self, payload):
        """Use the answer callback or ask a person; the workflow owns response policy."""
        import json

        if self.answer_callback is not None:
            return self.answer_callback(payload)
        self.write(json.dumps(payload, indent=2))
        # Closing the prompt answers the workflow's cancellation contract; it
        # does not silently approve and is distinct from aborting an active run.
        approval = payload["kind"] == "approval"
        try:
            return self.read(
                "approve / reject / cancel: "
                if approval
                else "Answer (/cancel to abandon): "
            )
        except (EOFError, KeyboardInterrupt):
            return "cancel" if approval else payload.get("cancel", "/cancel")
