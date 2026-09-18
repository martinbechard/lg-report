"""Group trace capture, normalization, pricing, and report export code.

Saved run data is the common boundary for HTML and Excel. Capture observes agent
execution; exporters consume that evidence without rerunning the agent. Missing
usage and unknown prices must remain distinguishable from zero cost.

AI attribution: Generated with AI assistance.

Copyright (c) 2026 Martin.Bechard@DevConsult.ca
"""
