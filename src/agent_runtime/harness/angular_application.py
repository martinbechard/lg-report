"""Prepare the selected sample and start the Angular application's HTTP listener.

This application owns web startup settings. The server owns HTTP handlers and
browser conversation lifetimes; it creates workflows when browser sessions start.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""


class AngularApplication:
    """Launch the browser-facing application using the shared catalog and CLI args."""

    def __init__(self, catalog, args):
        """Retain startup inputs until run() prepares the selected sample."""
        self.catalog = catalog
        self.args = args

    def run(self, sample_id):
        """Prepare listener defaults; browser requests own subsequent conversations."""
        from agent_runtime.web.server import start_workflow_api_listener

        from .settings import prepare_sample

        settings, options, prompts = prepare_sample(self.catalog, sample_id, self.args)
        start_workflow_api_listener(
            catalog=self.catalog,
            sample_id=sample_id,
            settings=settings,
            port=self.args.port,
            options=options,
            prompts=prompts,
            env_file=self.args.env_file,
        )
