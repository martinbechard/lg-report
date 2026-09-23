"""Group normalization, pricing, report export, and the reporting command.

Saved run data is the common boundary for HTML and Excel. Runtime capture lives
in agent_runtime.harness; exporters consume its evidence without rerunning agents.
Missing usage and unknown prices must remain distinguishable from zero cost.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""


def main():
    """Load the reporting CLI only when the installed lg-report command runs."""
    from .cli import main as cli_main

    cli_main()
