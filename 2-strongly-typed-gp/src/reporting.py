# reporting.py - printing results, statistics, and the computational effort metric

import math
import numpy as np
from config import SUCCESS_THRESHOLD


# PER-RUN AND MULTI-RUN SUMMARIES

def print_run_table(results):
    """Print a per-run summary table."""
    hdr = (f"{'Run':>3}  {'Seed':>5}  {'SuccGen':>7}  "
           f"{'FinalAdj':>9}  {'FinalRaw':>9}  "
           f"{'Time(s)':>7}  {'TestMAE':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        sg  = str(r["success_gen"]) if r["success_gen"] is not None else "-"
        print(f"{r['run_idx']+1:>3}  {r['seed']:>5}  {sg:>7}  "
              f"{r['best'].adjusted_fitness:>9.5f}  {r['best_raw']:>9.5f}  "
              f"{r['elapsed']:>7.1f}  {r['test_mae']:>10.6f}")


def multi_run_stats(results):
    """Return a dict of aggregate statistics across all runs."""
    raws      = [r["best_raw"]              for r in results]
    train_mae = [r["best_raw"] / 500        for r in results]
    adj       = [r["best"].adjusted_fitness for r in results]
    test_mae  = [r["test_mae"]              for r in results]
    times     = [r["elapsed"]              for r in results]
    successes = [r for r in results if r["success_gen"] is not None]

    return {
        "mean_raw":       np.mean(raws),
        "std_raw":        np.std(raws, ddof=1),
        "best_raw":       min(raws),
        "mean_train_mae": np.mean(train_mae),
        "std_train_mae":  np.std(train_mae, ddof=1),
        "best_train_mae": min(train_mae),
        "mean_adj":       np.mean(adj),
        "std_adj":        np.std(adj, ddof=1),
        "best_adj":       max(adj),
        "n_successes":    len(successes),
        "n_runs":         len(results),
        "mean_test_mae":  np.mean(test_mae),
        "std_test_mae":   np.std(test_mae, ddof=1),
        "best_test_mae":  min(test_mae),
        "mean_time":      np.mean(times),
        "total_time":     sum(times),
    }


def print_summary(stats):
    """Print multi-run summary table."""
    tr = 0.146  # target range kW (A1 report: max - min = 0.177 - 0.031)
    print("\n" + "="*60)
    print("  MULTI-RUN RESULTS SUMMARY")
    print("="*60)
    rows = [
        ("Raw fitness (sum|error|, 500 cases)",
         f"{stats['mean_raw']:.5f}", f"{stats['std_raw']:.5f}", f"{stats['best_raw']:.5f}"),
        ("Train MAE per case (kW)",
         f"{stats['mean_train_mae']:.6f}", f"{stats['std_train_mae']:.6f}",
         f"{stats['best_train_mae']:.6f}"),
        ("Train MAE as % of target range",
         f"{stats['mean_train_mae']/tr*100:.2f}%",
         f"{stats['std_train_mae']/tr*100:.2f}%",
         f"{stats['best_train_mae']/tr*100:.2f}%"),
        ("Adjusted fitness 1/(1+raw)",
         f"{stats['mean_adj']:.5f}", f"{stats['std_adj']:.5f}", f"{stats['best_adj']:.5f}"),
        ("Test MAE (kW)",
         f"{stats['mean_test_mae']:.6f}", f"{stats['std_test_mae']:.6f}",
         f"{stats['best_test_mae']:.6f}"),
        ("Test MAE as % of target range",
         f"{stats['mean_test_mae']/tr*100:.2f}%",
         f"{stats['std_test_mae']/tr*100:.2f}%",
         f"{stats['best_test_mae']/tr*100:.2f}%"),
    ]
    print(f"  {'Metric':<42} {'Mean':>12} {'StdDev':>12} {'Best':>12}")
    print("  " + "-" * 80)
    for row in rows:
        print(f"  {row[0]:<42} {row[1]:>12} {row[2]:>12} {row[3]:>12}")
    print(f"\n  Successes (adj >= {SUCCESS_THRESHOLD}): {stats['n_successes']} / {stats['n_runs']}")
    print(f"  Mean time per run: {stats['mean_time']:.1f} s")
    print(f"  Total time: {stats['total_time']:.1f} s ({stats['total_time']/60:.1f} min)")


# COMPUTATIONAL EFFORT (Koza 1992)

def computational_effort(results, pop_size, z=0.99):
    """
    Compute the Koza (1992) computational effort metric.

    For each generation i we compute:
        p(i)  = fraction of runs that found a solution by generation i
            (empirical success probability for one run)

        R(z, M, i) = ceil(log(1-z) / log(1-p(i)))
            = minimum number of independent runs needed so that the
            probability of at least one success >= z

        I(z, M, i) = R(z, M, i) * M * (i + 1)
            = total individuals evaluated across those runs and generations

    Find the generation i* that minimises I - the computational "sweet spot"
    where running multiple shorter runs beats running fewer longer ones.
    """
    max_gen = max(r["n_gen"] for r in results)
    n_runs  = len(results)

    # For each generation, count cumulative successes
    successes_by_gen = []
    cum = 0
    for g in range(max_gen):
        cum += sum(1 for r in results if r["success_gen"] == g)
        successes_by_gen.append(cum)

    best_I   = float("inf")
    best_gen = 0

    for i, n_suc in enumerate(successes_by_gen):
        p = n_suc / n_runs
        if p <= 0 or p >= 1:
            continue
        R = math.ceil(math.log(1 - z) / math.log(1 - p))
        I = R * pop_size * (i + 1)
        if I < best_I:
            best_I   = I
            best_gen = i

    return best_gen, best_I


# A1 vs A2 COMPARISON

# A1 results (20 runs, seeds 42-61):
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
}


def print_comparison(a2_stats):
    """Print a side-by-side A1 vs A2 comparison table."""
    tr = 0.146
    print("\n" + "="*60)
    print("  COMPARISON: A1 (Canonical GP) vs A2 (STGP)")
    print("="*60)
    hdr = f"  {'Metric':<40} {'A1':>12} {'A2':>12} {'Delta':>10}"
    print(hdr)
    print("  " + "-" * 76)

    def row(label, a1_val, a2_val, fmt=".6f", pct=False):
        delta = a2_val - a1_val
        if pct:
            print(f"  {label:<40} {a1_val/tr*100:>11.2f}% {a2_val/tr*100:>11.2f}% "
                  f"{delta/tr*100:>+9.2f}%")
        else:
            print(f"  {label:<40} {a1_val:>{12}{fmt}} {a2_val:>{12}{fmt}} "
                  f"{delta:>{10}{fmt}}")

    row("Mean Train MAE (kW)",        A1_RESULTS["mean_train_mae"], a2_stats["mean_train_mae"])
    row("Best Train MAE (kW)",        A1_RESULTS["best_train_mae"], a2_stats["best_train_mae"])
    row("Mean Test MAE (kW)",         A1_RESULTS["mean_test_mae"],  a2_stats["mean_test_mae"])
    row("Best Test MAE (kW)",         A1_RESULTS["best_test_mae"],  a2_stats["best_test_mae"])
    row("Mean Test MAE (% of range)", A1_RESULTS["mean_test_mae"],  a2_stats["mean_test_mae"], pct=True)

    a1_sr = A1_RESULTS["n_successes"] / A1_RESULTS["n_runs"]
    a2_sr = a2_stats["n_successes"]   / a2_stats["n_runs"]
    print(f"  {'Success rate':<40} {a1_sr:>12.1%} {a2_sr:>12.1%} "
          f"{(a2_sr-a1_sr):>+10.1%}")
    print(f"  {'Mean time per run (s)':<40} {A1_RESULTS['mean_time']:>12.1f} "
          f"{a2_stats['mean_time']:>12.1f} "
          f"{a2_stats['mean_time']-A1_RESULTS['mean_time']:>+10.1f}")
