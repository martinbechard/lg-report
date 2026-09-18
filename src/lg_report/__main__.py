"""Run the reporting CLI when invoked as python -m lg_report.

Delegate to the same handler used by the installed lg-report command so the two
entry points cannot drift. This module starts argument parsing when executed;
it does not define a separate sample runner.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from .report.cli import main

main()
