"""Create deterministic, ASCII-friendly slugs from text.

This module's responsibility is the small public ``slugify`` function; it has
no service, filesystem, or third-party dependencies. The function assumes its
input is a Python string and deliberately raises ``TypeError`` otherwise.
AI assistance: implementation drafted with AI assistance for this exercise.
Copyright: generated for this exercise; no separate copyright holder claimed.
"""

import unicodedata


def slugify(value: str) -> str:
    """Return a lowercase, hyphen-separated slug for *value*.

    Accented Latin characters are decomposed and reduced to their ASCII base
    characters. Characters that cannot contribute to an ASCII slug act as
    separators when they are punctuation or whitespace and are otherwise
    ignored.
    """
    if not isinstance(value, str):
        raise TypeError("slugify() argument must be a string")

    normalized = unicodedata.normalize("NFKD", value)
    parts = []
    separator_pending = False

    for character in normalized:
        # Combining marks carry accents already represented by the preceding
        # base character; discarding them preserves the ASCII base character.
        if unicodedata.combining(character):
            continue

        if character.isascii() and character.isalnum():
            if separator_pending and parts:
                parts.append("-")
            parts.append(character.lower())
            separator_pending = False
        elif character.isspace() or unicodedata.category(character).startswith(
            ("P", "S")
        ):
            # Delay adding the hyphen so leading, repeated, and trailing
            # separators cannot appear in the returned slug.
            if parts:
                separator_pending = True

    return "".join(parts)
