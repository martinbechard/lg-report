"""Behavioral tests for the deterministic public ``slugify`` function.

This module's responsibility is to verify the documented input/output
contract without services or third-party dependencies. Tests assume the public
function is imported from ``slug`` and do not inspect its implementation.
AI assistance: tests drafted with AI assistance for this exercise.
Copyright: generated for this exercise; no separate copyright holder claimed.
"""

import unittest

from slug import slugify


class SlugifyTests(unittest.TestCase):
    def test_normal_text_becomes_a_lowercase_slug(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")

    def test_whitespace_and_punctuation_runs_collapse(self):
        value = "  one\t--\n two...three / four  "
        self.assertEqual(slugify(value), "one-two-three-four")

    def test_accented_and_unicode_text_is_normalized(self):
        self.assertEqual(slugify("Café déjà vu"), "cafe-deja-vu")

    def test_repeated_and_boundary_separators_are_removed(self):
        self.assertEqual(slugify("---already--separated---"), "already-separated")

    def test_empty_and_punctuation_only_input_returns_empty_string(self):
        self.assertEqual(slugify(""), "")
        self.assertEqual(slugify("!? — …"), "")

    def test_non_string_values_raise_type_error(self):
        for value in (None, 42, ["text"], b"text"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    slugify(value)

    def test_whitespace_symbols_and_combining_marks_only_return_empty(self):
        self.assertEqual(slugify(" \t\n\r"), "")
        self.assertEqual(slugify("!@#$%^&*()[]{}<>"), "")
        self.assertEqual(slugify("\u0301\u0327\u20dd"), "")

    def test_long_mixed_separator_runs_and_boundaries_collapse(self):
        value = "  /\\—…___---  First  !@#$  second  ***  "
        self.assertEqual(slugify(value), "first-second")

    def test_non_decomposing_unicode_is_discarded(self):
        self.assertEqual(slugify("漢字 Ελληνικά привет"), "")
        self.assertEqual(slugify("start漢字end"), "startend")

    def test_digits_and_alphanumeric_text_are_preserved(self):
        self.assertEqual(slugify("Release 2024 - Version 2B"), "release-2024-version-2b")

    def test_filtering_allows_empty_output_between_separators(self):
        self.assertEqual(slugify("---漢字***"), "")

    def test_mixed_case_accents_and_digits_share_normalized_boundaries(self):
        value = "  -- ÉxAMPLe... 007 -- "
        self.assertEqual(slugify(value), "example-007")

    def test_unsupported_letters_are_discarded_without_new_separators(self):
        self.assertEqual(slugify("left漢字right"), "leftright")


if __name__ == "__main__":
    unittest.main()
