"""Create deterministic ASCII slugs for local document titles.

Accents with Latin decomposition are removed; characters without ASCII
representation are dropped. The function has no filesystem or network effects.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import re
import unicodedata


def slugify(title: str) -> str:
    """Return a lowercase, hyphen-separated ASCII form of a title."""
    if not isinstance(title, str):
        raise TypeError("title must be a string")
    # Decompose Latin accents before the ASCII boundary; unsupported scripts
    # are dropped rather than transliterated through an undeclared dependency.
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")
