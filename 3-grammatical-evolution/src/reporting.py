#!/usr/bin/env python3
# reporting.py - statistics, summaries, and A1/A2/A3 comparison

import numpy as np
from config import N_FITNESS_CASES


# PRIOR RESULTS FOR CROSS-ASSIGNMENT COMPARISON

# A1: Canonical GP - 20 runs, seeds 42-61, N_FITNESS_CASES=500
A1_RESULTS = {
    "mean_train_mae": 0.012770,
    "std_train_mae":  0.000437,
    "best_train_mae": 0.012139,
    "mean_test_mae":  0.018123,
    "std_test_mae":   0.000350,
    "best_test_mae":  0.017664,
    "n_successes":    4,
    "n_runs":         20,
    "mean_time":      14.6,
    "label":          "A1 (Canonical GP)",
}

# A2: Strongly Typed GP - 20 runs, seeds 42-61, N_FITNESS_CASES=500
A2_RESULTS = {
    "mean_train_mae": 0.012522,
    "std_train_mae":  0.000358,
    "best_train_mae": 0.011939,
    "mean_test_mae":  0.018027,
    "std_test_mae":   0.000251,
    "best_test_mae":  0.017672,
    "n_successes":    9,
    "n_runs":         20,
    "mean_time":      41.6,
    "label":          "A2 (STGP)",
}


def print_run_table(results):
    """Print a per-run summary table."""
    hdr = (f"{'Run':>3}  {'Seed':>5}  {'FinalAdj':>9}  "
           f"{'FinalRaw':>10}  {'TrainMAE':>9}  {'Time(s)':>7}  {'TestMAE':>10}")
    print("\n" + hdr)
    for r in results:
        train_mae = r["best_raw"] / N_FITNESS_CASES
        print(f"{r['run_idx']+1:>3}  {r['seed']:>5}  "
              f"{r['best'].adjusted_fitness:>9.5f}  "
              f"{r['best_raw']:>10.4f}  "
              f"{train_mae:>9.6f}  "
              f"{r['elapsed']:>7.1f}  "
              f"{r['test_mae']:>10.6f}")


def multi_run_stats(results):
    """Return aggregate statistics across all runs."""
    valid_results = [r for r in results if r["test_mae"] != float("inf")]

    raws      = [r["best_raw"]              for r in results]
    train_mae = [r["best_raw"] / N_FITNESS_CASES for r in results]
    adj       = [r["best"].adjusted_fitness for r in results]
    test_mae  = [r["test_mae"]              for r in valid_results] if valid_results else [0.0]
    times     = [r["elapsed"]              for r in results]

    return {
        "mean_raw":       np.mean(raws),
        "std_raw":        np.std(raws, ddof=1) if len(raws) > 1 else 0.0,
        "best_raw":       min(raws),
        "mean_train_mae": np.mean(train_mae),
        "std_train_mae":  np.std(train_mae, ddof=1) if len(train_mae) > 1 else 0.0,
        "best_train_mae": min(train_mae),
        "mean_adj":       np.mean(adj),
        "std_adj":        np.std(adj, ddof=1) if len(adj) > 1 else 0.0,
        "best_adj":       max(adj),
        "mean_test_mae":  np.mean(test_mae),
        "std_test_mae":   np.std(test_mae, ddof=1) if len(test_mae) > 1 else 0.0,
        "best_test_mae":  min(test_mae),
        "n_runs":         len(results),
        "mean_time":      np.mean(times),
        "total_time":     sum(times),
    }


def print_summary(stats):
    """Print multi-run summary table."""
    tr = 0.146   # kW target range (max 0.177 - min 0.031)
    print("\n  MULTI-RUN RESULTS SUMMARY  (A3: Structure-Based GE)\n")
    rows = [
        ("Raw fitness (Σ|error|, 1500 cases)",
         f"{stats['mean_raw']:.4f}", f"{stats['std_raw']:.4f}", f"{stats['best_raw']:.4f}"),
        ("Train MAE per fitness case (kW)",
         f"{stats['mean_train_mae']:.6f}", f"{stats['std_train_mae']:.6f}",
         f"{stats['best_train_mae']:.6f}"),
        ("Train MAE as % of target range",
         f"{stats['mean_train_mae']/tr*100:.2f}%",
         f"{stats['std_train_mae']/tr*100:.2f}%",
         f"{stats['best_train_mae']/tr*100:.2f}%"),
        ("Adjusted fitness 1/(1+raw)",
         f"{stats['mean_adj']:.5f}", f"{stats['std_adj']:.5f}", f"{stats['best_adj']:.5f}"),
        ("Test MAE (kW, full held-out set)",
         f"{stats['mean_test_mae']:.6f}", f"{stats['std_test_mae']:.6f}",
         f"{stats['best_test_mae']:.6f}"),
        ("Test MAE as % of target range",
         f"{stats['mean_test_mae']/tr*100:.2f}%",
         f"{stats['std_test_mae']/tr*100:.2f}%",
         f"{stats['best_test_mae']/tr*100:.2f}%"),
    ]
    print(f"  {'Metric':<44} {'Mean':>12} {'StdDev':>12} {'Best':>12}")
    for row in rows:
        print(f"  {row[0]:<44} {row[1]:>12} {row[2]:>12} {row[3]:>12}")
    print(f"\n  Runs: {stats['n_runs']}")
    print(f"  Mean time per run: {stats['mean_time']:.1f} s")
    print(f"  Total time: {stats['total_time']:.1f} s ({stats['total_time']/60:.1f} min)")


def print_comparison(a3_stats):
    """Print a three-way A1 vs A2 vs A3 comparison table."""
    tr = 0.146
    print("\n  COMPARISON: A1 (Canonical GP)  vs  A2 (STGP)  vs  A3 (Structure-Based GE)\n")

    hdr = f"  {'Metric':<38} {'A1':>12} {'A2':>12} {'A3':>12}"
    print(hdr)

    def row(label, a1_val, a2_val, a3_val, fmt=".6f"):
        print(f"  {label:<38} {a1_val:>{12}{fmt}} {a2_val:>{12}{fmt}} {a3_val:>{12}{fmt}}")

    def row_pct(label, a1_val, a2_val, a3_val):
        print(f"  {label:<38} {a1_val/tr*100:>11.2f}% {a2_val/tr*100:>11.2f}% "
              f"{a3_val/tr*100:>11.2f}%")

    row("Mean Train MAE (kW)",
        A1_RESULTS["mean_train_mae"], A2_RESULTS["mean_train_mae"], a3_stats["mean_train_mae"])
    row("Best Train MAE (kW)",
        A1_RESULTS["best_train_mae"], A2_RESULTS["best_train_mae"], a3_stats["best_train_mae"])
    row("Mean Test MAE (kW)",
        A1_RESULTS["mean_test_mae"],  A2_RESULTS["mean_test_mae"],  a3_stats["mean_test_mae"])
    row("Best Test MAE (kW)",
        A1_RESULTS["best_test_mae"],  A2_RESULTS["best_test_mae"],  a3_stats["best_test_mae"])
    row_pct("Mean Test MAE (% of range)",
            A1_RESULTS["mean_test_mae"], A2_RESULTS["mean_test_mae"], a3_stats["mean_test_mae"])
    print(f"  {'Runs':<38} {A1_RESULTS['n_runs']:>12} {A2_RESULTS['n_runs']:>12}"
          f" {a3_stats['n_runs']:>12}")
    print(f"  {'Mean time per run (s)':<38} {A1_RESULTS['mean_time']:>12.1f}"
          f" {A2_RESULTS['mean_time']:>12.1f} {a3_stats['mean_time']:>12.1f}")
