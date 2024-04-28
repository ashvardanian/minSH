from enum import Enum
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Literal, Optional
from time import perf_counter
from collections import defaultdict
from glob import glob
import math
import json
from typing import Callable, Tuple, Any
import random

import fire
import pandas as pd
from tqdm import tqdm
import numpy as np

from astar import h_dijkstra, align, build_seedh, build_seedh_for_pruning, build_straighest_zeroline_heuristic


class AlgorithmType(Enum):
    WAGNER_FISCHER = "Wagner-Fischer"
    DIJKSTRA = "Dijkstra"
    SEED = "Seed"
    SEED_PRUNING = "Seed Pruning"
    ZERO_STRAIGHTLINE = "Zero Straightline"


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


def wagner_fisher(s1: str, s2: str) -> Result:
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


def wrapped_dijkstra(A: str, B: str) -> Callable[[Tuple[int, int]], int]:
    return h_dijkstra


def wrapped_seed(A: str, B: str) -> Any:
    k = math.ceil(math.log(len(A), 4))
    return build_seedh(A, B, k)


def wrapped_seed_prune(A: str, B: str):
    k = math.ceil(math.log(len(A), 4))
    return build_seedh_for_pruning(A, B, k)

def wrapped_zero_straightline(A: str, B: str):
    return build_straighest_zeroline_heuristic(A, B)


def main(
    path: str,
    split: Literal["line", "whitespace"] = "line",
    jobs: Optional[int] = None,
    max_time: Optional[float] = None,
):
    """Benchmarking script for the A* algorithm with different heuristics.

    :param path:    Path to the newline- or whitespace-delimited dataset file, or a GLOB pattern like `data/*.txt`
                    if you want to benchmark multiple datasets.
    :param split:   Tokenization method to split the dataset into strings. Either "line" or "whitespace".
    :param jobs:    Number of parallel string cells to perform. If not specified, all strings possible
                    pairs of strings from files will be evaluates.

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
        # > dataset_content = Str(File(dataset)) faster
        with open(dataset, "r") as f:
            dataset_content = f.read()
        dataset_tokenized = (
            dataset_content.splitlines() if split == "line" else dataset_content.split()
        )
        dataset_tokenized = [s for s in dataset_tokenized if len(s) > 5]

        # Random sample pairs from strings
        strings_a = (
            random.sample(dataset_tokenized, jobs) if jobs else dataset_tokenized
        )
        strings_b = (
            random.sample(dataset_tokenized, jobs) if jobs else dataset_tokenized
        )
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
            start_time = perf_counter()
            result = wagner_fisher(a, b)
            end_time = perf_counter()
            results_per_algo[algo].append(
                BenchmarkResult(
                    preprocessing_time=0,
                    run_time=end_time - start_time,
                    cells=result.cells,
                    distance=result.distance,
                    length_a=len(a),
                    length_b=len(b),
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
            (wrapped_dijkstra, AlgorithmType.DIJKSTRA),
            (wrapped_seed, AlgorithmType.SEED),
            (wrapped_seed_prune, AlgorithmType.SEED_PRUNING),
            (wrapped_zero_straightline, AlgorithmType.ZERO_STRAIGHTLINE),
        ]:
            print(f"-- Running algorithm: {algo}")

            algo_run_time = 0
            for a, b in tqdm(strings_pairs):
                prep_time = perf_counter()
                heuristic = heursitic_generator(a, b)
                start_time = perf_counter()
                _, distance, cells = align(a, b, heuristic)
                end_time = perf_counter()
                results_per_algo[algo].append(
                    BenchmarkResult(
                        preprocessing_time=start_time - prep_time,
                        run_time=end_time - start_time,
                        cells=cells,
                        distance=distance,
                        length_a=len(a),
                        length_b=len(b),
                    )
                )

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
                        "Algorithm": algo.value,
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


if __name__ == "__main__":
    fire.Fire(main)
