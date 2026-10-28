# reporting.py - printing results, statistics, and the computational effort metric

import math
import os
import platform
import statistics
import sys
from datetime import datetime

import numpy as np
from config import N_FITNESS_CASES, POP_SIZE, SUCCESS_THRESHOLD


def log_environment():
    """
    Print system info so results can be reproduced on the same machine.
    """
    print(f"Python {sys.version.split()[0]}  "
          f"NumPy {np.__version__}  "
          f"{platform.system()}  "
          f"{datetime.now():%Y-%m-%d %H:%M}  "
          f"PID={os.getpid()}")


def print_config(cfg):
    """
    Print a summary of the active configuration values.
    """
    print(f"Pop={cfg.POP_SIZE}  Gens={cfg.MAX_GENERATIONS}  Runs={cfg.N_RUNS}  Seed={cfg.BASE_SEED}")
    print(f"Terminals={cfg.TERMINAL_SET}")
    print(f"Functions={list(cfg.FUNCTION_SET)}\n")


def print_statistics(results, baselines):
    """
    Print summary statistics across all N_RUNS runs.
    """
    raws      = [r["best_raw"]  for r in results]
    adjs      = [1 / (1 + r)   for r in raws]
    times     = [r["elapsed"]   for r in results]
    test_maes = [r["test_mae"]  for r in results]
    n         = len(results)
    sd        = lambda v: statistics.stdev(v) if n > 1 else 0.0

    print("\n" + "═"*60)
    print("  MULTI-RUN STATISTICS")
    print("═"*60)
    print(f"  Runs        : {n}")

    # Raw fitness: total absolute error summed over N_FITNESS_CASES cases
    print(f"  Raw fitness   mean={statistics.mean(raws):.5f}  "
          f"std={sd(raws):.5f}  best={min(raws):.5f}")

    # Adjusted fitness: 1/(1+raw) — closer to 1.0 is better
    print(f"  Adj fitness   mean={statistics.mean(adjs):.5f}  "
          f"std={sd(adjs):.5f}  best={max(adjs):.5f}  "
          f"(success >= {SUCCESS_THRESHOLD})")

    print(f"  Time/run(s)   mean={statistics.mean(times):.1f}  total={sum(times):.1f}")
    print(f"  Successes   : {sum(1 for r in results if r['success_gen'] is not None)}/{n}")

    # Test-set MAE: evaluated on the full held-out test set
    print(f"  Test MAE(kW)  mean={statistics.mean(test_maes):.6f}  "
          f"std={sd(test_maes):.6f}  best={min(test_maes):.6f}")

    # Identify the best run by lowest raw fitness
    best_run = min(results, key=lambda r: r["best_raw"])
    print(f"\n  Best individual (run {best_run['run_idx'] + 1}, seed={best_run['seed']}):")
    print(f"  {best_run['best'].tree.to_lisp()}")
    print("═"*60)

    # Baseline comparison on the test set
    print(f"\n  Baselines (test set, {baselines['n_test']:,} rows):")
    print(f"    Mean predictor     MAE = {baselines['mean_mae']:.6f} kW")
    print(f"    Persistence (x_1)  MAE = {baselines['persistence_mae']:.6f} kW")
    print(f"    GP mean            MAE = {statistics.mean(test_maes):.6f} kW")
    print(f"    GP best            MAE = {min(test_maes):.6f} kW")

    # Per-generation table for the best run
    print(f"\n  Per-generation best  (run {best_run['run_idx'] + 1})")
    print(f"  {'Gen':>4}  {'Norm':>10}  {'Raw':>10}  {'Size':>5}  {'Depth':>5}")
    for g in best_run["best_per_gen"]:
        print(f"  {g['gen']:4d}  {g['norm']:10.6f}  {g['raw']:10.5f}"
              f"  {g['size']:5d}  {g['depth']:5d}")


def computational_effort(results, z=0.99):
    """
    Compute the Koza (1992) computational effort metric.

    For each generation i we compute:
        p(i)  = fraction of runs that found a solution by generation i
            (empirical success probability for one run)

        R(z,M,i) = ceil( log(1-z) / log(1-p(i)) )
            = minimum number of independent runs needed so that the
            probability of at least one success >= z

        I(z,M,i) = R(z,M,i) * M * i
            = total individuals evaluated across those runs and generations

    Find the generation i* that minimises I - the computational "sweet spot"
    where running multiple shorter runs beats running fewer longer ones.
    """
    n_runs  = len(results)
    max_gen = max(r["n_gen"] for r in results)

    print(f"\n{'─'*60}\n  COMPUTATIONAL EFFORT  z={z:.0%}  M={POP_SIZE}\n{'─'*60}")

    best_effort, best_gen = math.inf, -1

    for gen in range(1, max_gen + 1):
        # Count how many runs had found a solution strictly before generation gen
        successes = sum(
            1 for r in results
            if r["success_gen"] is not None and r["success_gen"] < gen
        )
        p = successes / n_runs
        if p <= 0:
            continue   # no successes yet — can't compute R

        p = min(p, 1 - 1e-12)   # avoid log(0) when p == 1.0
        R      = math.ceil(math.log(1 - z) / math.log(1 - p))
        effort = R * POP_SIZE * gen

        if effort < best_effort:
            best_effort, best_gen = effort, gen

    if best_effort == math.inf:
        print(f"  No successes (threshold={SUCCESS_THRESHOLD}). Cannot compute effort.")
        print(f"  Tip: lower SUCCESS_THRESHOLD or increase MAX_GENERATIONS.")
    else:
        print(f"  Min I(0.99, {POP_SIZE}, i) = {best_effort:,d}  at generation {best_gen}")

    print("─"*60)
