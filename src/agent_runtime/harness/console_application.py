"""Own console sample selection without putting workflow lifecycle in a client.

The catalog lists and constructs samples. ConsoleClient handles slash commands, reads
messages, displays responses, and answers interruptions. This application owns
the resulting sample launches. Switching samples ends the current
conversation before its graph, checkpoint, and resources are replaced.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from .console_client import ConsoleClient


class ConsoleApplication:
    """Own terminal launches: one static run or an interactive sample-selection loop."""

    def __init__(self, catalog, args, *, read=input, write=print):
        self.catalog = catalog
        self.args = args
        self.read = read
        self.write = write

    def run_session(self, selected, client):
        """Prepare the selection and execute it with this console's existing client.

        Initial sample overrides do not leak when the user selects another
        workflow. /new repeats the current selection with a fresh conversation.
        This named method owns the console-specific launch policy.
        """
        from copy import copy

        from .configure_sample_script import configure_sample_script
        from .execute_conversation import execute_conversation
        from .settings import prepare_sample

        args = copy(self.args)
        if selected != self.args.sample:
            args.options = {}
            args.source = args.target = args.mode = None
            args.values = args.request = args.out = None
        settings, options, prompts = prepare_sample(self.catalog, selected, args)
        # Input mode and model mode are independent: authored prompts can also
        # drive real models. Static mode supplies interruption answers as well;
        # structured interactive lessons keep their human-answer behavior.
        if (
            args.client == "static"
            or self.catalog.get(selected).interaction
            or prompts is not None
        ):
            configure_sample_script(
                client,
                self.catalog,
                selected,
                prompts=prompts,
                script_answers=args.client == "static",
                scenario=args.scenario,
                decision=args.decision,
            )
        execute_conversation(
            catalog=self.catalog,
            sample_id=selected,
            settings=settings,
            options=options,
            client=client,
            env_file=args.env_file,
            show_context=args.show_context,
            public_trace=args.public_trace,
        )

    def run(self, sample_id):
        """Run static input once; interactive input keeps the sample menu open.

        Both modes use run_session for preparation, optional script setup, and
        execution. Only interactive mode needs catalog commands and switching.
        """
        if self.args.client == "static":
            client = ConsoleClient(read=self.read, write=self.write)
            self.run_session(sample_id, client)
            return
        while sample_id is not None:
            sample = self.catalog.get(sample_id)
            self.write(
                f"Sample: {sample.name} ({sample.id}) — /samples, /sample ID, /new, /quit"
            )
            client = ConsoleClient(
                catalog=self.catalog,
                sample_id=sample_id,
                read=self.read,
                write=self.write,
            )
            self.run_session(sample_id, client)
            sample_id = client.next_sample
