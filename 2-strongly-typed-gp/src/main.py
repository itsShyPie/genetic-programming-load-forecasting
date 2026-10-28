#!/usr/bin/env python3
# main.py - entry point

from config import DATA_PATH, N_RUNS, POP_SIZE
from data import load_dataset, build_test_set
from runner import run_gp
from reporting import (
    print_run_table,
    multi_run_stats,
    print_summary,
    computational_effort,
    print_comparison,
)


def main():
    # Load full dataset into memory as plain list of floats
    print("Loading dataset ...")
    loads = load_dataset(DATA_PATH)
    print(f"  {len(loads):,} readings loaded")

    # Build held-out test set once (shared across all runs)
    X_test, Y_test = build_test_set(loads)
    print(f"  Test set: {len(Y_test):,} rows")

    # Run GP N_RUNS times; each run gets a different seed
    results = []
    for run_idx in range(N_RUNS):
        r = run_gp(run_idx, loads, X_test, Y_test)
        results.append(r)

    # Print per-run table, then mean/std/best across runs
    print_run_table(results)

    stats = multi_run_stats(results)
    print_summary(stats)

    # Print best individual tree found across runs
    best_run = min(results, key=lambda r: r["best_raw"])
    print(f"\n  Best individual (run {best_run['run_idx']+1}, "
          f"seed {best_run['seed']}):")
    print(f"  {best_run['best'].tree.to_lisp()}")

    # Compute the Koza (1992) I(z, M, i) metric
    gen_star, I_star = computational_effort(results, POP_SIZE)
    print(f"\n  Computational effort: I* = {I_star:,} at generation {gen_star}")

    # Side-by-side comparison with A1 results
    print_comparison(stats)


if __name__ == "__main__":
    main()
