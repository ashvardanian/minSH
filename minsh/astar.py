import sys, math
from heapq import heappush, heappop  # Heap for the priority queue
from collections import defaultdict
from typing import Union, Callable

import numpy as np  # To compute sum[i] = num[i] + sum[i+1]
from fenwick import FenwickTree  # To add and remove matches

from minsh.rolling_hashes import rolling_hash_string_pair_intersection
from minsh.utils import ceildiv, read_fasta_file, print_stats

h_dijkstra = lambda ij: 0  # Dijkstra's dummy heuristic


def build_seedh(A, B, k):
    """Builds the admissible seed heuristic for strings A and B with k-mers."""
    seeds = [A[i : i + k] for i in range(0, len(A) - k + 1, k)]  # O(n)
    kmers = {
        B[j : j + k] for j in range(len(B) - k + 1)
    }  # O(nk), O(n) with rolling hash (Rabin-Karp)
    is_seed_missing = [s not in kmers for s in seeds] + [False] * 2  # O(n)
    suffix_sum = np.cumsum(is_seed_missing[::-1])[::-1]  # O(n)
    return lambda ij, k=k: suffix_sum[ceildiv(ij[0], k)]  # O(1)


def build_seedh_for_pruning(A, B, k):
    S = [A[i : i + k] for i in range(0, len(A) - k + 1, k)]
    K = defaultdict(set)
    [K[B[j : j + k]].add(j) for j in range(len(B) - k + 1)]
    M = [K[s] for s in range(len(S))]
    misses = FenwickTree(len(S) + 2)
    misses.init([not js for js in M] + [0] * 2)

    return lambda ij, k=k, M=M, misses=misses: misses.range_sum(
        ceildiv(ij[0], k), len(misses)
    )


def build_straighest_zeroline_heuristic(A, B):
    """
    Build the heuristic for the A* algorithm that gives a lower bound where if you are at (i, j) it assumes that you
    take a straight line down with slope -1 and then go straight right or down to the end. Which one you do depends on the
    values of i and j. If you are further to the right you will go down and if you are further down you will go right.
    How far you go is the distance of the smaller of those two.
    """

    def logic(x: int, y: int, x_max: int, y_max: int) -> int:
        assert (
            x_max + 1 >= x and y_max + 1 >= y
        ), f"({x}, {y}) is not in the grid ({x_max}, {y_max})"
        if x == x_max + 1 or y == y_max + 1:
            return x_max + y_max + 1 + 1  # Super bad never go here

        dx = x_max - x
        dy = y_max - y
        # If we are equidistant (on the diagonal) then we will go diagnal => 0
        # If we are closer to the bottom then we will go right => dx > dy & then you use up dy first and then dx - dy going right
        # If we are closer to the right then we will go down => dy > dx & then you use up dx first and then dy - dx going down
        return abs(dx - dy)

    return lambda ij: logic(ij[0], ij[1], len(A), len(B))


def build_straighest_zeroline_max_substring_heuristic(
    A, B, r=10, max_bin_search_iters=1000
):
    """
    Build a heuristic that gives the tightest possible global lower bound akin to `wrapped_straighest_zeroline_heuristic`.
    The change here is that we find the length of the longest substring that is common to both strings. This corresponds to
    the longest diagonal of only zeros. Any diagonal that we consider must have at most this many zeros. Moreover, we know that
    it is never beneficial to take diagonals that are (this value / 2) orthogonal distance (counting cells) away from the main
    diagonal(s).

    For strings that are as if they were drawn from an IID distribution per-coordinate over an alphabet, we should expect w.h.p.
    the length of the longest shared substring is somewhere close to O(log(|A| + |B|)), or generally very sublinear. This is because
    if the alphabet has size |C|, then the probability of matching a substring of length k is |C|^(-k). The expected number of matching
    substrings of this length is at most (|A| + |B|)|C|^(-k) by linearity of expectation. This means that if we take
    k = log_|C|(|A| + |B|) = O(log(|A| + |B|)), then the expected number of matching substrings of this length is 1. We can scale k to
    get a very low expected value very easily (k -> rk => |C|^(-r) expected number of matching substrings => an exponential improvement
    expected value (we want zero) in r). We can scale k by a slow-moving function of |A| + |B| and the chernoff bound to get a w.h.p.
    proof that the longest substring should almost always O(log(n)).

    TODO(Adriano) find a bound for how far those matches => no longer a relevant match. Use something like this to narrow down
    which substring lengths are valid in what ranges.
    """

    k = 0

    # Pursue a simple stragy:
    # 1. Try to guess that the longest substring is at most length r * log(|A| + |B|)
    # 2. If we are unlucky (very unlikely), binary search
    # => Most of the time O(n) and worst case O(log(n)) using a robin-karp rolling hash strategy to find whether or not there are
    #    substrings of a certain length. For a given length,
    #
    # log_|C|(|A| + |B|) = log(|A| + |B|) / log(|C|)
    c = len(set(A) | set(B))
    # For now, just support ASCII (because of the rolling hash dummy above)
    # Comment out if necessary for your use case (but you may need to edit the rolling hash code)
    assert c <= 2**8
    assert c <= 2**16

    # Hack to deal with byte strings
    if isinstance(A, bytes):
        A = A.decode("utf-8")
    if isinstance(B, bytes):
        B = B.decode("utf-8")

    k = None
    k_tentative = min(
        r * int(math.ceil(math.log(len(A) + len(B)) / math.log(c))),
        max(0, min(len(A), len(B)) - 1),
    )
    has_substring_length_k_tentative = rolling_hash_string_pair_intersection(
        A, B, k_tentative
    )
    if has_substring_length_k_tentative:
        # Do binary search and set an optimal value to k
        # Lower bound is inclusive, top bound is exclusive
        k_max = min(len(A), len(B)) + 1
        k_min = k_tentative
        assert k_max > k_min

        largest_possible_substring_length = k_max
        bin_search_iters = 0
        # Works because of monotonicity of inclusion of shared substring because if substrings size n are shared
        # then the prefixes (or suffixes or whatver) of those substrings cover the lower sizes.
        while k_max - k_min > 0:
            # We found it
            if k_max - k_min == 1:
                k = k_min
                break

            # We went too long
            if bin_search_iters > max_bin_search_iters:
                k = largest_possible_substring_length
                break
            bin_search_iters += 1

            # Try something in the middle
            k_probe = (k_max + k_min) // 2
            has_substring_length_k_probe = rolling_hash_string_pair_intersection(
                A, B, k_probe
            )
            if has_substring_length_k_probe:
                # Inclusive
                k_min = k_probe
            else:
                # Exclusive
                k_max = k_probe
                largest_possible_substring_length = k_max
            assert k_max > k_min
    else:
        k = k_tentative
    assert k is not None
    assert 0 <= k and k <= min(len(A), len(B))
    assert isinstance(k, int)

    # The actual function for the heuristic
    def logic(x: int, y: int, x_max: int, y_max: int, k: int) -> int:
        assert (
            x_max + 1 >= x and y_max + 1 >= y
        ), f"({x}, {y}) is not in the grid ({x_max}, {y_max})"

        # These are not in the graph so avoid them (due to a quirk of astar.py this is necessary)
        #
        # By providing a phony value of k we can effectively say "don't go here"
        if x == x_max + 1 or y == y_max + 1:
            return x_max + y_max + 1 + 1

        # There are |A| + |B| - 1 diagonal indices: one for each row (left) and then one for each column (top), but we double-count the
        # bottom left.
        # NOTE that "x" is the row and "y" is the column, so we have it slightly flipped here.
        #
        # The middle diagonals are the ones that touch the top left corner and the bottom right corner. There are at most 2.
        # The one at the top left corner is the one where x = 0 and y = 0 => x - y = 0
        # The one at the bottom right corner is the one where x = |A| and y = |B| => x - y = |A| - |B|
        # When you shift by one diagonal you add one or two to the value (x, y) -> x - y therefore if we look at x - y
        #   we can infer if we are between the two diagonals by seeing if we are between the two values.
        starter_diagonal_index = 0
        ender_diagonal_index = x_max - y_max
        diagonal_index = x - y

        # If we are away from the nearest diagonal and outside the center band (of the rectangle grid) by more than
        # ceil(k / 2  + 1) cells then it is never worth to be here, because the center diagonal is cheaper (it takes |A| + |B| steps, whereas this
        # took us k + 1 to get out here and go back plus the k + 1 to go down the diagonal; however, we could have gotten to that same place in
        # the same exactly by going down the center diagonal. This is true for all higher/lower diagonals.
        #
        # By providing a phony value of k we can effectively say "don't go here"
        orthogonal_distance = int(math.ceil(float(k / 2 + 1)))
        max_diagonal_index = (
            max(starter_diagonal_index, ender_diagonal_index) + orthogonal_distance
        )
        min_diagonal_index = (
            min(max_diagonal_index, ender_diagonal_index) - orthogonal_distance
        )
        if diagonal_index > max_diagonal_index or diagonal_index < min_diagonal_index:
            return x_max + y_max + 1 + 1

        dx = x_max - x
        dy = y_max - y

        # Imagine doing the minimum of dx and dy first, then doing the rest of the distance on one of the axes; you can see that
        # this visualization has the same cost as anything else, and it is necessarily the case that this many diagonals must be taken,
        # because otherwise you could do better by taking one more diagonal. You will take as many zero diagonals as you can.
        horizontal_distance = abs(dx - dy)
        diagonal_distance = min(dx, dy)
        diagonal_distance = max(0, diagonal_distance - k)
        return diagonal_distance + horizontal_distance

    return lambda ij: logic(ij[0], ij[1], len(A), len(B), k)


def next_states_with_cost(u, A, B):
    """Suggest the next state, given the current state and the strings A and B.

    :param u:   current state - a tuple of two integers, for the current position in A and B.
    :param A:   string A.
    :param B:   string B.
    :return:    three tuples, each containing the next state and the cost of the transition
                for the right, down, and diagonal moves with cost 0 for match, 1 otherwise.
    """
    return (
        ((u[0] + 1, u[1]), 1),
        ((u[0], u[1] + 1), 1),
        ((u[0] + 1, u[1] + 1), A[u[0]] != B[u[1]]),
    )


def align(
    A: Union[str, bytes],
    B: Union[str, bytes],
    h: Callable,
    return_stats: bool = False,
):
    """Standard A* on the grid A x B using a given heuristic h.

    :param A:   string A.
    :param B:   string B.
    :param h:   heuristic function `h(ij) -> int`, where `ij` is a tuple of two integers.
    :param return_stats:    whether to return the statistics of the alignment.

    :return:    by default, return the dictionary of state distances.
                if `return_stats` is True, return a  tuple with the dictionary of state distances,
                distance to target, and number of cells evaluated during traversal.
    """
    start = (0, 0)  # Start state
    target = (len(A), len(B))  # Target state
    Q = []  # Priority queue with candidate states
    heappush(Q, (0, start))  # Push start state with priority 0
    g = {start: 0}  # Cost of getting to each state
    cells = 0

    # Barrier to avoid index out of bounds
    A += b"!" if isinstance(A, bytes) else "!"
    B += b"!" if isinstance(B, bytes) else "!"

    while Q:
        _, u = heappop(Q)  # Pop state u with lowest priority
        if u == target:
            return (
                (
                    g,  # costs dictionary
                    g[(len(A) - 1, len(B) - 1)],  # distance from A to B
                    cells,  # number of matrix cells evaluated
                )
                if return_stats
                else g
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

        cells += 1
        for v, edit_cost in next_states_with_cost(u, A, B):  # For all edges u->v
            new_cost_to_next = g[u] + edit_cost  # Try optimal path through u->v
            if v not in g or new_cost_to_next < g[v]:  # If new path is better
                g[v] = new_cost_to_next  # Update cost to v
                priority = new_cost_to_next + h(v)  # Compute priority
                heappush(Q, (priority, v))  # Push v with new priority


def main():
    if len(sys.argv) != 3:
        print("Usage: python astar.py <A.fa> <A.fa>")
    else:
        A, B = map(read_fasta_file, sys.argv[1:3])
        k = math.ceil(math.log(len(A), 4))
        h_seed = build_seedh(A, B, k)

        g_seed = align(A, B, h_seed)
        print_stats(A, B, k, g_seed)


if __name__ == "__main__":
    main()
