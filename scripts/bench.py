from enum import Enum
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Literal, Optional
from time import perf_counter
from collections import defaultdict
from glob import glob
import math
import random
import argparse

import pandas as pd
from tqdm import tqdm
import numpy as np

from minsh.astar import (
    h_dijkstra,
    align,
    build_seedh,
    build_seedh_for_pruning,
    build_straighest_zeroline_heuristic,
    build_straighest_zeroline_max_substring_heuristic,
)


class AlgorithmType(Enum):
    WAGNER_FISCHER = "Wagner-Fischer"
    DIJKSTRA = "Dijkstra"
    SEED = "Seed"
    SEED_PRUNING = "Seed Pruning"
    MULTI_K = "Multi-k"
    ZERO_STRAIGHTLINE = "Zero Straightline"
    ZERO_STRAIGHTLINE_MAX_SUBSTRING = "Zero Straightline Max Substring"


@dataclass
class Result:
    matrix: np.ndarray
    distance: int
    cells: int


@dataclass
class BenchmarkResult:
    preprocessing_time: float
    run_time: float
    cells: int
    distance: int
    length_a: int
    length_b: int


def wagner_fisher(s1, s2) -> Result:
    # Create a matrix of size (len(s1)+1) x (len(s2)+1)
    matrix = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)

    # Initialize the first column and first row of the matrix
    for i in range(len(s1) + 1):
        matrix[i, 0] = i
    for j in range(len(s2) + 1):
        matrix[0, j] = j

    # Compute Levenshtein distance
    cells = 0
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            substitution_cost = s1[i - 1] != s2[j - 1]
            matrix[i, j] = min(
                matrix[i - 1, j] + 1,  # Deletion
                matrix[i, j - 1] + 1,  # Insertion
                matrix[i - 1, j - 1] + substitution_cost,  # Substitution
            )
            cells += 1

    # Return the Levenshtein distance
    return Result(matrix, matrix[len(s1), len(s2)], cells)


def wrapped_dijkstra(A, B):
    return h_dijkstra


def alphabet_size(s):
    return len(set(c for c in s))


def wrapped_seed(A, B):
    s = max(alphabet_size(A), alphabet_size(B))
    k = math.ceil(math.log(len(A), s))
    return build_seedh(A, B, k)


def wrapped_seed_prune(A, B):
    s = max(alphabet_size(A), alphabet_size(B))
    k = math.ceil(math.log(len(A), s))
    return build_seedh_for_pruning(A, B, k)


def wrapped_multi_k(A, B):
    s = max(alphabet_size(A), alphabet_size(B))
    k = math.ceil(math.log(len(A), s))
    h_value = np.full(len(A) + 2, 0)
    for j in range(2, k + 1):
        seeds = [A[i : i + j] for i in range(0, len(A) - j + 1, j)]  # O(n)
        kmers = {
            B[j : j + j] for j in range(len(B) - j + 1)
        }  # O(nk), O(n) with rolling hash (Rabin-Karp)
        is_seed_missing = [s not in kmers for s in seeds] + [False] * 2  # O(n)
        suffix_sum = np.cumsum(is_seed_missing[::-1])[::-1]  # O(n)
        # This is expanding suffix_sum so that the lookup
        # suffix_sum[ceildiv(ij[0], k)] becomes h_value[ij[0]].
        local_h_value = np.concatenate(([suffix_sum[0]], np.repeat(suffix_sum[1:], j)))[
            : len(A) + 2
        ]
        h_value = np.maximum(h_value, local_h_value)
    return lambda ij, k=k: h_value[ij[0]]  # O(1)


def wrapped_straighest_zeroline_heuristic(A, B):
    return build_straighest_zeroline_heuristic(A, B)


def wrapped_straighest_zeroline_max_substring_heuristic(A, B):
    # Use derault r, etc...
    return build_straighest_zeroline_max_substring_heuristic(A, B)


def main(
    path: str,
    split: Literal["line", "whitespace"] = "line",
    jobs: Optional[int] = None,
    max_time: Optional[float] = None,
    min_length: Optional[int] = None,
):
    """Benchmarking script for the A* algorithm with different heuristics.

    :param path:        Path to the newline- or whitespace-delimited dataset file, or a GLOB pattern like `data/*.txt`
                        if you want to benchmark multiple datasets. Datasets can be downloaded from
                        https://github.com/rghilduta/human-chromosome-data-generator/tree/main/examples . The protein
                        dataset was generated using ./generate_chromosome_data.sh -r 50000 -l 1000
    :param split:       Tokenization method to split the dataset into strings. Either "line" or "whitespace".
    :param jobs:        Number of parallel string cells to perform. If not specified, all strings possible
                        pairs of strings from files will be evaluates.
    :param max_time:    Maximum time in seconds to spend on each algorithm. If not specified, all strings will be evaluated.
    :param min_length:  Minimum length of strings to consider. If not specified, all strings will be evaluated.
    """
    assert split in [
        "line",
        "whitespace",
    ], "Invalid split method. Use 'line' or 'whitespace'."
    datasets = glob(path) if "*" in path else [path]
    max_time = float(max_time) if max_time else None

    for dataset in datasets:
        print(f"- Running dataset: {dataset}")

        # Prepare a contain to assemble the results for the current dataset
        results_per_algo: Dict[AlgorithmType, List[BenchmarkResult]] = defaultdict(list)

        # Load the dataset and split it into whitespace or newline separated strings
        with open(dataset, "r") as f:
            text = f.read(1_000_000 * 64)
        tokens = text.splitlines() if split == "line" else text.split()
        if min_length:
            tokens = [s for s in tokens if len(s) > min_length]
        mean_length = np.mean([len(s.encode("utf8")) for s in tokens])
        print(f"-- Average token: {mean_length:.2f} bytes")

        # Random sample pairs from strings
        strings_a = random.sample(tokens, jobs) if jobs else tokens
        strings_b = random.sample(tokens, jobs) if jobs else tokens
        strings_pairs = (
            list(zip(strings_a, strings_b))
            if jobs
            else list(product(strings_a, strings_b))
        )

        # Run the baseline algo, aggregating all the results for the Wagner Fisher
        print(f"-- Running algorithm: Wagner-Fisher")
        algo = AlgorithmType.WAGNER_FISCHER
        algo_run_time = 0
        for a, b in tqdm(strings_pairs):
            # Normalize strings to be byte-string, whould be more efficient
            a_binary = a.encode("utf8")
            b_binary = b.encode("utf8")
            start_time = perf_counter()
            result = wagner_fisher(a_binary, b_binary)
            end_time = perf_counter()
            results_per_algo[algo].append(
                BenchmarkResult(
                    preprocessing_time=0,
                    run_time=end_time - start_time,
                    cells=result.cells,
                    distance=result.distance,
                    length_a=len(a_binary),
                    length_b=len(b_binary),
                )
            )

            # Don't waste too much time on bad algos ;)
            algo_run_time += end_time - start_time
            if max_time and algo_run_time > max_time:
                break

        # Log the number of evaluated cells as opposed to the total product of all lengths
        cells_eval = sum(r.cells for r in results_per_algo[algo])
        cells_totall = sum(r.length_a * r.length_b for r in results_per_algo[algo])
        print(
            f"--- {cells_eval:,} / {cells_totall:,} cells = {cells_eval / cells_totall:.2%}"
        )

        for heursitic_generator, algo in [
            (wrapped_straighest_zeroline_heuristic, AlgorithmType.ZERO_STRAIGHTLINE),
            (
                wrapped_straighest_zeroline_max_substring_heuristic,
                AlgorithmType.ZERO_STRAIGHTLINE_MAX_SUBSTRING,
            ),
            (wrapped_dijkstra, AlgorithmType.DIJKSTRA),
            (wrapped_seed, AlgorithmType.SEED),
            (wrapped_seed_prune, AlgorithmType.SEED_PRUNING),
            (wrapped_multi_k, AlgorithmType.MULTI_K),
        ]:
            print(f"-- Running algorithm: {algo.name}")

            algo_run_time = 0
            for a, b in tqdm(strings_pairs):
                # Normalize strings to be byte-string, whould be more efficient
                a_binary = a.encode("utf8")
                b_binary = b.encode("utf8")
                prep_time = perf_counter()
                heuristic = heursitic_generator(a_binary, b_binary)
                start_time = perf_counter()
                states, distance, cells = align(
                    a_binary,
                    b_binary,
                    heuristic,
                    return_stats=True,
                )
                end_time = perf_counter()
                results_per_algo[algo].append(
                    BenchmarkResult(
                        preprocessing_time=start_time - prep_time,
                        run_time=end_time - start_time,
                        cells=cells,
                        distance=distance,
                        length_a=len(a_binary),
                        length_b=len(b_binary),
                    )
                )

                del states

                # Don't waste too much time on bad algos ;)
                algo_run_time += (end_time - start_time) + (start_time - prep_time)
                if max_time and algo_run_time > max_time:
                    break

            # Log the number of evaluated cells as opposed to the total product of all lengths
            cells_eval = sum(r.cells for r in results_per_algo[algo])
            cells_totall = sum(r.length_a * r.length_b for r in results_per_algo[algo])
            print(
                f"--- {cells_eval:,} / {cells_totall:,} cells = {cells_eval / cells_totall:.2%}"
            )

        # Print the results, save every result in a separate `.csv`
        aggregated_results = []
        for algo, results in results_per_algo.items():
            for result in results:
                aggregated_results.append(
                    {
                        "Algorithm": algo.name,
                        "Preprocessing Time": result.preprocessing_time,
                        "Run Time": result.run_time,
                        "Cells": result.cells,
                        "Distance": result.distance,
                        "Length A": result.length_a,
                        "Length B": result.length_b,
                    }
                )
        df = pd.DataFrame(aggregated_results)
        df.to_csv(f"{dataset}.csv", index=False)
        summary_statistics = df.groupby("Algorithm").agg(
            {
                "Run Time": ["sum", "mean", "var"],
                "Cells": ["mean", "var"],
                "Distance": ["mean", "var"],
            }
        )
        print(summary_statistics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmarking for sequence alignment algorithms."
    )
    parser.add_argument(
        "path", type=str, help="Path to the dataset file or GLOB pattern"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="line",
        choices=["line", "whitespace"],
        help="Method to split dataset into strings",
    )
    parser.add_argument(
        "--jobs", type=int, help="Number of parallel string alignments to perform"
    )
    parser.add_argument(
        "--max-time",
        type=float,
        help="Maximum time in seconds to spend on each algorithm",
    )
    parser.add_argument(
        "--min-length", type=int, help="Minimum length of strings to consider"
    )
    args = parser.parse_args()
    main(args.path, args.split, args.jobs, args.max_time, args.min_length)
