"""Select console, static, or web startup through one shared command-line app.

The package __main__ delegates here. Argument parsing, sample preparation, and
conversation execution remain in their own modules; this module selects the
launch path without embedding the console session loop or HTTP handlers.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

from .argument_parser import parse_arguments


def main():
    """One command-line application lists, selects, and runs every discovered sample."""
    from .console_application import ConsoleApplication
    from .sample_catalog import SampleCatalog

    catalog = SampleCatalog()
    parser, args = parse_arguments(catalog)
    if args.list:
        for sample in catalog.samples.values():
            print(f"{sample.id}: {sample.name} — {sample.description}")
        return
    try:
        if args.client == "angular":
            from .angular_application import AngularApplication

            AngularApplication(catalog, args).run(args.sample)
        else:
            ConsoleApplication(catalog, args).run(args.sample)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
