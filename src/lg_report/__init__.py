"""Expose the installed lg-report command while keeping package import lightweight.

The CLI implementation is imported only when main is called. Consumers can import
agent, platform, or report modules without also starting argument parsing or a
workflow. The console-script entry point is declared in pyproject.toml. This
initializer exposes no global runtime, credentials, or persistent state.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""


def main():
    """Enter the reporting CLI used by the installed lg-report console command.

    Delay CLI imports until execution so importing this package for its data models
    does not initialize the command's dependencies. Argument handling and exit
    behavior belong to cli.main; this wrapper defines no separate command policy.
    Exceptions and exit behavior propagate unchanged so ``python -m`` and the
    installed console script report the same outcome.
    """
    from .report.cli import main as cli_main

    cli_main()
