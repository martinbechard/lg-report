#!/bin/sh
# Summarize the prepared fictional order data and save a small text report.
# Run from the sample workspace. Fail on missing input or failed output so the
# agent receives a real nonzero exit instead of a misleading success message.
# AI attribution: Generated with AI assistance by Northstar.
# Copyright (c) 2026 Martin.Bechard@DevConsult.ca
set -eu
awk -F, 'NR > 1 { orders++; units += $2 } END { printf "Orders: %d\nTotal units: %d\n", orders, units }' orders.csv > summary.txt
cat summary.txt
