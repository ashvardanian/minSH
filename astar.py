from __future__ import annotations
import sys, math
import numpy as np                  # To compute sum[i] = num[i] + sum[i+1]
from fenwick import FenwickTree     # To add and remove matches
from utils import *                 # Trivial helper functions
from typing import Callable, Tuple, Dict, List
from heapq import heappush, heappop  # Heap for the priority queue
from collections import defaultdict

import numpy as np  # To compute sum[i] = num[i] + sum[i+1]
from fenwick import FenwickTree  # To add and remove matches

from utils import ceildiv, read_fasta_file, print_stats

h_dijkstra = lambda ij: 1  # Dijkstra's dummy heuristic


def build_seedh(A: str, B: str, k: int) -> Callable[[int], int]:
    """Builds the admissible seed heuristic for A and B with k-mers.
    
    The way this works is pretty simple: 
    1. You imagine that we will match A TO B (note that this is symmetric since otherwise there would be a
        shorter from B to A: shortest in either direction is the same and there is a 1:1 mapping between change
        sequences in either direction by reversal).
    2. You look at ORDERED and DISJOINT substrings of A - the "seeds" - as well as
        the set of ALL substrings (of the corresponding lengths) in B - the "kmers."
    3. A seed that is not present in a kmer MUST be changed by at least one character. There are likely more
        restrictions but this is a simple first pass.
    4. Sum from end to front of A to count at any point in A how many remaining seeds will need to be changed
        (if we are at that point in A - that suffix of a - then everything that remains needs to be changed if it
        was not in B at all).

    """
    assert k <= len(A) and k <= len(B) # Don't allow this

    seeds = [ A[i:i+k] for i in range(0, len(A)-k+1, k) ]           # O(n)   
    kmers = { B[j:j+k] for j in range(len(B)-k+1) }                 # O(nk), O(n) with rolling hash (Rabin-Karp)
    is_seed_missing = [ s not in kmers for s in seeds ] + [False]*2 # O(n)
    suffix_sum = np.cumsum(is_seed_missing[::-1])[::-1]             # O(n)
    return lambda ij, k=k: suffix_sum[ ceildiv(ij[0], k) ]          # O(1)

def build_seedh_for_pruning(A: str, B: str, k: int) -> Callable[[int], int]:
    """
    Build something analogous to build_seedh but I think the fenwick tree lets you change cumulative sums.
    """
    S = [ A[i:i+k] for i in range(0, len(A)-k+1, k) ] # Seeds
    K = defaultdict(set); [ K[B[j:j+k]].add(j) for j in range(len(B) - k + 1) ] # Kmers dict
    M = [ K[s] for s in range(len(S)) ] # Mapping (by index) from seed to set of kmers that match (their indices)

    # Fenwick tree lets you quickly find the sum of ranges of an array AND edit them
    # (think: better than just splitting array in middle recursively)
    misses = FenwickTree(len(S)+2); misses.init([not js for js in M] + [0]*2)
    
    # Calculate the cumulative sum TO the END using the Fenwick tree
    return lambda ij, k=k, M=M, misses=misses: \
        misses.range_sum( ceildiv(ij[0], k), len(misses) )

def next_states_with_cost(u, A, B):
    """Generates three states following curr (right, down, diagonal) with cost 0
    for match, 1 otherwise.
    
    This is just a helper to specifically know which node in the graph to go to 
    and which cost to associate with it.
    """
    return [ ((u[0] + 1, u[1]    ), 1),
             ((u[0],     u[1] + 1), 1),
             ((u[0] + 1, u[1] + 1), A[u[0]] != B[u[1]]) ]

def align(A: str, B: str, h: Callable[[Tuple[int, int], int]]) -> Tuple[Dict[Tuple[int, int], int], Tuple[int, int], int]:
    """
    Standard A* on the grid A x B using a given heuristic h.

    Input two strings A and B and a heuristic h. Heuristic is a callable from node to
    expected distance from node to target. The heuristic MUST be admissible (never over-estimate).

    Return the DP grid, but in the form of a dictionary that only includes the visited nodes. NOTE that the
    graph is of the form where (i, j) points to (i+1, j), (i, j+1), and (i+1, j+1) with costs 1, 1, and 0 | 1 (depending
    on whether there is a character match). It is a DAG.

    A* Algorithm works like this:
        - Maintain a heap queue of states whose priority is based on their "expected distance" (i.e. inversely proportional)
        - At each step pop the state with the lowest expected distance and go to its neighbors (push them onto the heap)
        (repeat...)
    Admissible heuristic guarantees that once we reach the ending node, we are done.

    There is an additional feature here which is pruning: TODO(Adriano) understand the pruning well.
    """
    start: Tuple[int, int] = (0, 0)              # Start state
    target: Tuple[int, int] = (len(A), len(B))   # Target state
    
    Q: List[Tuple[int, Tuple[int, int]]] = []                      # Priority queue with candidate states
    heappush(Q, (0, start))     # Push start state with priority 0
    g: Dict[Tuple[int, int], int] = { start: 0 }            # Cost of getting to each state
    
    A += '!'; B += '!'          # Barrier to avoid index out of bounds

    comparisons: int = 0 # Count amount of "work" done

    while Q:
        _, u = heappop(Q)  # Pop state u with lowest priority
        if u == target:
            return (
                g,  # costs dictionary
                g[(len(A) - 1, len(B) - 1)],  # distance from A to B
                comparisons,  # number of matrix cells evaluated
            )

        if u[0] > target[0] or u[1] > target[1]:
            continue  # Skip states after target

        if hasattr(h, "misses"):  # If the heuristic supports pruning
            if not u[0] % h.k:  # If expanding at the beginning of a seed
                s = u[0] // h.k
                if u[1] in h.M[s]:  # If the expanded state is a beginning of a match
                    h.M.remove(s, u[1])  # Remove match from M
                    assert len(h.M[s]) >= 0
                    # If no more matches for this seed, then increase the misses
                    if not h.M[s]:
                        assert not h.misses[s]
                        h.misses.add(s, +1)

        for v, edit_cost in next_states_with_cost(u, A, B):  # For all edges u->v
            new_cost_to_next = g[u] + edit_cost  # Try optimal path through u->v
            if v not in g or new_cost_to_next < g[v]:  # If new path is better
                g[v] = new_cost_to_next  # Update cost to v
                priority = new_cost_to_next + h(v)  # Compute priority
                heappush(Q, (priority, v))  # Push v with new priority
            comparisons += 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python astar.py <A.fa> <A.fa>")
    else:
        # 1. Read the two files
        # 2. Pick a reasonable k
        # 3. Align it
        # 4. Print information about how many nodes were visited in the graph and the edit distance
        A, B   = map(read_fasta_file, sys.argv[1:3])
        k      = math.ceil(math.log(len(A), 4))
        h_seed = build_seedh(A, B, k)
        g, _, __      = align(A, B, h_seed)
        print_stats(A, B, k, g)
