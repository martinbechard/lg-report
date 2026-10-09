"""Check the public slug contract with standard-library unittest.

The workflow executes this suite before independent source review.
AI attribution: Generated with AI assistance by Northstar.
Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

import unittest

from slug import slugify


class SlugTests(unittest.TestCase):
    """Exercise distinct behaviors named in the plan."""

    def test_normal_title(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_runs_of_separators(self):
        self.assertEqual(slugify("Many   spaces... and punctuation!"), "many-spaces-and-punctuation")

    def test_latin_accent(self):
        self.assertEqual(slugify("Café Étude"), "cafe-etude")

    def test_empty(self):
        self.assertEqual(slugify(""), "")


    def test_only_separators(self):
        self.assertEqual(slugify("...  ---"), "")

    def test_outer_punctuation(self):
        self.assertEqual(slugify("!! Great Day ??"), "great-day")

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            slugify(None)
