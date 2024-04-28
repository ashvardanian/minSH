from __future__ import annotations

# This is a class of heuristic functions that are meant to be used in the A* algorithm to try and optimize
#
# Our current heuristic ideas fall into the following categories
# 1. Changing the constant k and optimizing it based on statistical testing for
#   the specific data strings we are going to be working with.
# 2. 