# Implementation Plan

Scope: build a small, deterministic, service-independent Python slug function. The public API is `slugify`, imported from `slug`. Only `/slug.py` and `/test_slug.py` may be changed for implementation work; this plan is the coordination record.

Contract decisions:
- `slugify(value)` accepts a string and returns a lowercase URL/path-safe slug.
- Unicode text is normalized to ASCII where possible, then lowercased.
- Runs of whitespace and punctuation become one hyphen; leading/trailing hyphens are removed.
- Empty or punctuation-only input returns `""`.
- Non-string input raises `TypeError` rather than being silently coerced.
- The implementation uses only the Python standard library and no external services or hidden state.
- Tests will be written but not run by this workflow.

## SLUG-001 — Implement `slugify` in `/slug.py`
Status: complete
Assigned: worker
Evidence: Independent reviewer approved the implementation. `/slug.py` exports deterministic standard-library-only `slugify`, handles NFKD normalization, ASCII alphanumeric output, delayed separator collapsing, trimming, empty results, and `TypeError` for non-strings. Tests were not run.

Implementation steps:
1. Read any existing `/slug.py` before editing and preserve unrelated content.
2. Implement the documented public function with a small, inspectable standard-library-only design.
3. Include a plain-language module header covering purpose, responsibilities, assumptions, AI assistance, and copyright; explain any non-obvious normalization or filtering branch with focused comments.
4. Cover representative behavior such as `"Hello, World!" -> "hello-world"`, repeated separators, accented text, and empty/punctuation-only input.
5. Handle the specified type error deterministically and avoid broad exception swallowing.

Acceptance criteria:
- `/slug.py` exports callable `slugify`.
- Outputs satisfy the contract above, including deterministic Unicode transliteration/normalization, separator collapsing, trimming, and empty results.
- Non-string values raise `TypeError`.
- No network, filesystem, service, or third-party dependency is required.
- The independent reviewer inspects the changed source and either approves it or returns this task for repair; reviewer approval is required before this task is closed. Tests are written but not run.

## SLUG-002 — Write behavioral tests in `/test_slug.py`
Status: complete
Assigned: worker
Evidence: Independent reviewer approved `/test_slug.py`. The tests import the public `slugify` from `slug` and cover ordinary input, whitespace and punctuation collapsing, accented Unicode normalization, repeated and boundary separators, empty and punctuation-only input, and non-string values raising `TypeError`. Assertions use deterministic standard-library `unittest` behavior checks without service or filesystem dependencies. Tests were not run, so execution remains unverified.

Implementation steps:
1. Read the current `/slug.py` and existing `/test_slug.py` before editing.
2. Add focused tests for normal words, whitespace and punctuation collapsing, Unicode accents, repeated/trimmed separators, empty and punctuation-only strings, and the non-string `TypeError` contract.
3. Use readable external-behavior assertions that would catch plausible regressions without merely copying implementation branches.
4. Keep tests independent of services and compatible with the chosen test framework already present, or use the standard library if no framework exists.

Acceptance criteria:
- `/test_slug.py` exercises every documented contract boundary and at least one representative ordinary slug.
- Tests import the public function from `slug`.
- Tests are deterministic, service-independent, and do not require network access.
- The independent reviewer inspects the test changes and either approves them or returns this task for repair; reviewer approval is required before this task is closed. Tests are written but not run.

## SLUG-003 — Strengthen boundary tests in `/test_slug.py`
Status: complete
Assigned: worker
Evidence: Independent reviewer approved `/test_slug.py`. The file retains the previously approved ordinary, normalization, separator, empty-input, and invalid-type tests and adds focused coverage for whitespace/symbol/combining-mark-only inputs (`test_whitespace_symbols_and_combining_marks_only_return_empty`), mixed separator runs with separators at both boundaries (`test_long_mixed_separator_runs_and_boundaries_collapse`), non-decomposing Unicode and unsupported characters (`test_non_decomposing_unicode_is_discarded`), digit and alphanumeric preservation (`test_digits_and_alphanumeric_text_are_preserved`), and filtering to empty output (`test_filtering_allows_empty_output_between_separators`). Assertions target the public `slugify` function and deterministic contract results, use only `unittest`, and do not inspect implementation details or external resources. Source inspection indicates the acceptance criteria are met; tests were not run, so execution remains unverified as required by the workflow.

Implementation steps:
1. Read the current `/test_slug.py` and `/slug.py` before editing, then preserve the existing approved coverage while adding only boundary-focused behavioral tests.
2. Add deterministic tests for whitespace/symbol/combining-mark-only inputs, long mixed separator runs and separators at both boundaries, Unicode characters that do not decompose to ASCII, preservation of digits and alphanumeric text, and empty output after filtering; add further invalid-type coverage only where it strengthens the documented contract.
3. Use external-behavior assertions against the public `slugify` function, keeping tests standard-library-only and independent of filesystem, network, services, and implementation details.

Acceptance criteria:
- `/test_slug.py` retains all existing tests and adds readable regression coverage for the listed boundary cases.
- Expected results reflect the documented contract: lowercase ASCII alphanumerics, collapsed/trimmed hyphens, discarded unsupported characters, and `""` when no output characters remain.
- The added tests do not duplicate internal branches or depend on hidden state, and no tests are run by this workflow.
- The independent reviewer inspects the changed test file and either approves it or returns this task for repair; reviewer approval is required before this task is closed.

## SLUG-004 — Further strengthen boundary tests in `/test_slug.py`
Status: complete
Assigned: worker
Evidence: Independent reviewer approved the current `/test_slug.py`. The file preserves prior approved coverage and adds focused public-behavior tests for mixed-case accented text with digits and boundaries (`test_mixed_case_accents_and_digits_share_normalized_boundaries`) and unsupported non-decomposing characters discarded without introducing separators (`test_unsupported_letters_are_discarded_without_new_separators`). Review confirmed coverage of the requested empty/whitespace, punctuation/separator, normalization, filtering, digit, unsupported-Unicode, and non-string boundaries, with deterministic standard-library-only assertions and no implementation-detail or external-resource dependencies. Only `/test_slug.py` was changed for this task. Tests were written but not run, so execution remains unverified.

Implementation steps:
1. Read the current `/slug.py` and `/test_slug.py` before editing, preserving all previously approved tests and the documented contract.
2. Add only concrete behavioral regression cases that strengthen boundaries not already adequately distinguished, such as empty/whitespace-only values, punctuation and separator runs around text, mixed-case and accented text, unsupported non-decomposing Unicode, digits, and non-string inputs where useful.
3. Keep assertions against the public `slugify` import, readable and deterministic, using no services, filesystem state, network, or implementation-detail inspection.

Acceptance criteria:
- `/test_slug.py` retains existing coverage and adds focused boundary tests that can detect plausible regressions in trimming, collapsing, normalization, filtering, empty results, and type validation.
- Expected values match the established contract: lowercase ASCII alphanumerics, collapsed and trimmed hyphens, discarded unsupported characters, empty output when nothing remains, and `TypeError` for non-strings.
- Only `/test_slug.py` is changed for this task; tests are written but not run by this workflow.
- The independent reviewer must inspect the changed test file and either approve it or reject it with source-backed findings; reviewer approval is required before the planner records this task complete. Review and closeout are part of this task, not separate tasks.
