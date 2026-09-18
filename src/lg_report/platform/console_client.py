"""Read user prompts and explicit text attachments from the terminal.

Files are read only when the user attaches them; nothing grants the agent access
to the surrounding filesystem. No web server or extra UI dependencies are needed.
Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from pathlib import Path

from .conversation import Attachment, Request


class ConsoleClient:
    """Use /attach PATH before a prompt, /send for files alone, and /quit to end.

    Only UTF-8 text is supported in this first sample. read/write are injectable
    to test console behavior without a terminal or provider call.
    """

    def __init__(self, read=input, write=print):
        """Use terminal I/O by default; injected callables support automated tests.

        read receives the prompt label and returns one line. write displays one
        string. Attachments stay queued until a request is actually submitted.
        """
        self.read = read
        self.write = write
        self.files = []

    def receive(self) -> Request | None:
        """Collect commands until there is a prompt to send or the user finishes."""
        while True:
            try:
                line = self.read("User: ")
            except (EOFError, KeyboardInterrupt):
                # EOF/Ctrl-C at the prompt is a normal user exit. Returning lets
                # the recorder close spans and generate the session report.
                return None
            command = line.strip()
            # Explicit exit discards unsent attachments; attaching a file alone
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
        """Display the latest answer; the session retains the complete history."""
        # A workflow may pause before producing any message. Only display
        # content when the returned transcript actually has a final element.
        if result.get("messages"):
            self.write(f"Assistant: {result['messages'][-1].content}")
        # Interrupts require a resume protocol this client does not implement;
        # report the pause even if the workflow also returned earlier messages.
        if result.get("__interrupt__"):
            self.write("Agent paused for approval; this sample cannot resume it.")
