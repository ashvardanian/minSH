from __future__ import annotations
from typing import List, Tuple, Callable

# TODO(Adriano) finish this!
# This is a class of heuristic functions that are meant to be used in the A* algorithm to try and optimize
#
# Our current heuristic ideas fall into the following categories
# 1. Changing the constant k and optimizing it based on statistical testing for
#   the specific data strings we are going to be working with.
# 2. Trying to do graph augmentations, etc...

# The optimal path is the path that is straightest in some sense but there is a catch: it must also have the highest number of
# of zero diagonals...
# 
# Can we use the fact that there are 4 symbols?

def character_differences(i: int, j: int, s1: str, s2: str) -> int:
    """
    For each string look at how many time each letter shows up before each index.
    """

    # Only support DNA strings for now
    assert all(x in "ACGT" for x in s)
    letter2idx = {letter: idx for idx, letter in enumerate("ACGT")}
    counts = [0] * 4
    for idx in range(i):
        counts[letter2idx[s1[idx]]] += 1
        counts[letter2idx[s2[idx]]] += 1


    pass 