#!/usr/bin/env python3
# main.py - entry point

import time

import numpy as np

import config
from data import build_test_set, load_dataset
from reporting import computational_effort, log_environment, print_config, print_statistics
from runner import run_gp


def main():
    # Print environment info (Python version, NumPy, OS) for reproducibility
    log_environment()
    print_config(config)

    # Load the full dataset into memory as a plain list of floats
    loads = load_dataset(config.DATA_PATH)
    n     = len(loads)
    print(f"Dataset: {n:,} rows  "
          f"train={int(n * config.TRAIN_RATIO):,}  "
          f"test={n - int(n * config.TRAIN_RATIO):,}\n")

    # Build the held-out test set once (shared across all runs)
    X_test, Y_test = build_test_set(loads)

    # Baseline 1: Persistence - predict y(t) = x_1 (same time yesterday)
    persistence_mae = float(np.mean(np.abs(X_test["x_1"] - Y_test)))

    # Baseline 2: Mean predictor - predict y(t) = mean of the training targets
    offset     = config.N_DAYS_BACK * 96
    train_end  = int(len(loads) * config.TRAIN_RATIO)
    train_mean = float(np.mean(loads[offset:train_end]))
    mean_mae   = float(np.mean(np.abs(train_mean - Y_test)))

    baselines = {
        "persistence_mae": persistence_mae,
        "mean_mae":        mean_mae,
        "n_test":          len(Y_test),
    }

    # Run the GP config.N_RUNS times; each run gets a different seed
    t_total = time.perf_counter()
    results = [run_gp(i, loads, X_test, Y_test) for i in range(config.N_RUNS)]

    # Print mean/std/best across all runs + the best individual tree
    print_statistics(results, baselines)
    print(f"\nTotal: {time.perf_counter() - t_total:.1f}s")

    # Compute the Koza (1992) I(z,M,i) metric separately
    computational_effort(results)


if __name__ == "__main__":
    main()
