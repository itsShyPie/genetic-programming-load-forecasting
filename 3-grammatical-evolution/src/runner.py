#!/usr/bin/env python3
# runner.py - main GE evolutionary loop for a single run

import random
import time
import numpy as np

from config import (
    BASE_SEED, CROSSOVER_RATE, MAX_GENERATIONS,
    MUTATION_RATE, N_FITNESS_CASES, N_RUNS,
    POP_SIZE, REPRODUCTION_RATE,
)
from data import build_fitness_cases
from fitness import evaluate_population, tournament_select
from operators import crossover, mutate, reproduce
from population import random_population
from primitives import eval_tree


def run_ge(run_idx, loads, X_test, Y_test):
    """
    Execute one complete GE run and return a results dict.

    Control model: generational with elitism
        Each generation the whole population is replaced by a new one.
        The single best individual (highest adjusted fitness) always survives
        unchanged (elitism), so the best solution found is never lost.

    No early-exit success threshold is used.  A2 feedback noted that an easy
    threshold stopped runs at generation 0 or 1, preventing proper convergence.
    All MAX_GENERATIONS are always executed.

    Operator application:
        n_crossover    = round(POP_SIZE * CROSSOVER_RATE)    =  400
        n_mutation     = round(POP_SIZE * MUTATION_RATE)     =   75
        n_reproduction = POP_SIZE - n_crossover - n_mutation =   25
    """
    seed = BASE_SEED + run_idx
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    print(f"\n  RUN {run_idx + 1}/{N_RUNS}  seed={seed}\n")
    t0 = time.perf_counter()

    # Build fitness cases (fixed for all generations of this run)
    X, Y = build_fitness_cases(loads, N_FITNESS_CASES, rng)

    # Create initial population (random chromosomes)
    pop = random_population(POP_SIZE)

    # Precompute exact offspring counts from rates
    n_crossover    = round(POP_SIZE * CROSSOVER_RATE)
    n_mutation     = round(POP_SIZE * MUTATION_RATE)
    n_reproduction = POP_SIZE - n_crossover - n_mutation

    best_per_gen = []

    for gen in range(MAX_GENERATIONS):

        # EVALUATE
        evaluate_population(pop, X, Y)

        best = max(pop, key=lambda ind: ind.adjusted_fitness)
        best_per_gen.append({
            "gen":   gen,
            "adj":   best.adjusted_fitness,
            "raw":   best.raw_fitness,
            "valid": best.valid,
        })

        if gen % 10 == 0 or gen == MAX_GENERATIONS - 1:
            valid_count = sum(1 for ind in pop if ind.valid)
            print(f"  Gen {gen:3d}  adj={best.adjusted_fitness:.5f}"
                  f"  raw={best.raw_fitness:.4f}  valid={valid_count}/{POP_SIZE}")

        # BUILD NEXT GENERATION
        nxt = [best.copy()]   # elitism

        # Reproduction
        for _ in range(n_reproduction):
            if len(nxt) >= POP_SIZE:
                break
            nxt.append(reproduce(tournament_select(pop)))

        # Crossover
        for _ in range(n_crossover):
            if len(nxt) >= POP_SIZE:
                break
            nxt.append(crossover(tournament_select(pop), tournament_select(pop)))

        # Mutation
        for _ in range(n_mutation):
            if len(nxt) >= POP_SIZE:
                break
            nxt.append(mutate(tournament_select(pop)))

        pop = nxt[:POP_SIZE]

    # FINAL EVALUATION
    evaluate_population(pop, X, Y)
    best    = max(pop, key=lambda ind: ind.adjusted_fitness)
    elapsed = time.perf_counter() - t0

    # Evaluate best individual on held-out test set
    if best.valid and best.tree is not None:
        test_preds = eval_tree(best.tree, X_test)
        test_raw   = float(np.sum(np.abs(test_preds - Y_test)))
        test_mae   = test_raw / len(Y_test)
    else:
        test_raw = float("inf")
        test_mae = float("inf")

    print(f"  Done  adj={best.adjusted_fitness:.5f}  raw={best.raw_fitness:.4f}  {elapsed:.1f}s")
    print(f"  Test  raw={test_raw:.4f}  mae={test_mae:.6f} kW")
    if best.valid and best.tree:
        print(f"  Best: {best.tree.to_lisp()[:120]}")

    return {
        "run_idx":      run_idx,
        "seed":         seed,
        "best":         best,
        "best_raw":     best.raw_fitness,
        "best_per_gen": best_per_gen,
        "elapsed":      elapsed,
        "n_gen":        len(best_per_gen),
        "test_raw":     test_raw,
        "test_mae":     test_mae,
    }
