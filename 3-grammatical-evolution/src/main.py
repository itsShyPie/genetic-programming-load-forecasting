#!/usr/bin/env python3
# main.py - entry point for Structure-Based Grammatical Evolution

from config import DATA_PATH, N_RUNS, POP_SIZE
from data import load_dataset, build_test_set
from runner import run_ge
from reporting import print_run_table, multi_run_stats, print_summary, print_comparison


def main():
    print("  COS 710 Assignment 3: Structure-Based GE")

    print("\nLoading dataset ...")
    loads = load_dataset(DATA_PATH)
    print(f"  {len(loads):,} readings loaded")

    X_test, Y_test = build_test_set(loads)
    print(f"  Test set: {len(Y_test):,} rows")

    results = []
    for run_idx in range(N_RUNS):
        r = run_ge(run_idx, loads, X_test, Y_test)
        results.append(r)

    print_run_table(results)

    stats = multi_run_stats(results)
    print_summary(stats)

    best_run = min(results, key=lambda r: r["best_raw"])
    print(f"\n  Best individual (run {best_run['run_idx']+1}, seed {best_run['seed']}):")
    if best_run["best"].valid and best_run["best"].tree:
        print(f"  {best_run['best'].tree.to_lisp()}")
    else:
        print("  (invalid derivation)")

    print_comparison(stats)


if __name__ == "__main__":
    main()
