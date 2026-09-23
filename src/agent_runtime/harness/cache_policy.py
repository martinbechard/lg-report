"""Define the cache lifetime requested by this application's Anthropic adapters.

Five minutes is an explicit application choice shared by request configuration
and pricing of writes whose lifetime was not specified. It does not guarantee
that a provider caches a request; actual billing still uses reported usage.

Architecture and ownership: docs/chat-composition.md.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""

# This selects the request policy, not a guarantee that a provider caches every
# token. Billing must still use reported cache creation/read usage. Five minutes
# is the teaching app's chosen lifetime; the price table retains both TTL tariffs
# so imported usage with explicit one-hour writes can still be priced correctly.
CACHE_TTL = "5m"
