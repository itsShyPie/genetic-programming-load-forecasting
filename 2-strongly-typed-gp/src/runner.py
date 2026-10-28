# runner.py - the main GP loop for a single run

import random
import time
import numpy as np

from config import (
    BASE_SEED, CROSSOVER_RATE, MAX_GENERATIONS,
    MUTATION_RATE, N_FITNESS_CASES, N_RUNS,
    POP_SIZE, REPRODUCTION_RATE, SUCCESS_THRESHOLD,
)
from data import build_fitness_cases
from fitness import evaluate_population, fps
from operators import crossover, mutate, reproduce
from population import ramped_half_and_half
from primitives import eval_tree


def run_gp(run_idx, loads, X_test, Y_test):
    """
    Execute one complete GP run and return the results as a dict.

    Control model: Generational with elitism

        Each generation the entire population is replaced by a new one.

        The single best individual is always copied into the next generation
        unchanged (elitism), so the best-ever solution never gets lost.

    Termination: whichever comes first:
        1. MAX_GENERATIONS reached
        2. best adjusted_fitness >= SUCCESS_THRESHOLD (early success exit)

    Operator application - KEY FIX vs A1:
        A1 applied operators by drawing a random number per offspring, meaning
        the actual fraction produced by each operator was left to chance.
        A2 uses exact counts derived from the rates before filling the population:
            n_crossover    = round(POP_SIZE x CROSSOVER_RATE)    = 400
            n_mutation     = round(POP_SIZE x MUTATION_RATE)     =  50
            n_reproduction = POP_SIZE - n_crossover - n_mutation =  50

    Reproducibility: both Python's random and numpy's random are seeded with
    BASE_SEED + run_idx, so this exact run can always be re-run identically.
    """
    seed = BASE_SEED + run_idx
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)   # separate instance for dataset sampling

    print(f"\n{'─'*60}\n  RUN {run_idx + 1}/{N_RUNS}  seed={seed}\n{'─'*60}")
    t0 = time.perf_counter()

    # Build training examples for this run (fixed for all generations of this run)
    X, Y = build_fitness_cases(loads, N_FITNESS_CASES, rng)

    # Create initial population via Ramped Half-and-Half
    pop = ramped_half_and_half(POP_SIZE)

    best_per_gen = []   # record best individual's stats every generation
    success_gen  = None

    for gen in range(MAX_GENERATIONS):

        # STEP 1: EVALUATE EVERY INDIVIDUAL #
        # Set raw, standardised, adjusted, and normalised fitness on every ind
        evaluate_population(pop, X, Y)

        # The best individual this generation has the highest adjusted fitness
        best = max(pop, key=lambda ind: ind.adjusted_fitness)

        # Record stats for reporting later
        best_per_gen.append({
            "gen":   gen,
            "adj":   best.adjusted_fitness,
            "raw":   best.raw_fitness,
            "size":  best.size(),
            "depth": best.depth(),
        })

        # Print a progress line every 10 generations
        if gen % 10 == 0 or gen == MAX_GENERATIONS - 1:
            print(f"  Gen {gen:3d}  adj={best.adjusted_fitness:.5f}"
                  f"  raw={best.raw_fitness:.5f}  size={best.size()}")

        # STEP 2: CHECK TERMINATION
        if best.adjusted_fitness >= SUCCESS_THRESHOLD:
            success_gen = gen
            print(f"  [SUCCESS] gen {gen}  adj={best.adjusted_fitness:.4f}")
            break

        # STEP 3: BUILD THE NEXT GENERATION
        # Compute exact offspring counts so operator rates are applied precisely.
        n_crossover    = round(POP_SIZE * CROSSOVER_RATE)
        n_mutation     = round(POP_SIZE * MUTATION_RATE)
        n_reproduction = POP_SIZE - n_crossover - n_mutation

        nxt = [best.copy()]   # elitism: best individual always survives unchanged

        # Reproduction: select one parent and copy it unchanged
        for _ in range(n_reproduction):
            if len(nxt) >= POP_SIZE:
                break
            nxt.append(reproduce(fps(pop)))

        # Crossover: select two parents and swap a random subtree
        for _ in range(n_crossover):
            if len(nxt) >= POP_SIZE:
                break
            nxt.append(crossover(fps(pop), fps(pop)))

        # Mutation: select one parent and replace a random subtree
        for _ in range(n_mutation):
            if len(nxt) >= POP_SIZE:
                break
            nxt.append(mutate(fps(pop)))

        pop = nxt[:POP_SIZE]   # trim to exact size (elitism may cause off-by-one)

    # FINAL EVALUATION
    # Evaluate last generation (loop above exits before evaluating if success)
    evaluate_population(pop, X, Y)
    best    = max(pop, key=lambda ind: ind.adjusted_fitness)
    elapsed = time.perf_counter() - t0

    # Evaluate best individual on the held-out test set (no side-effects on stored fitness)
    test_preds = eval_tree(best.tree, X_test)
    test_raw   = float(np.sum(np.abs(test_preds - Y_test)))
    test_mae   = test_raw / len(Y_test)

    print(f"  Done  adj={best.adjusted_fitness:.5f}  raw={best.raw_fitness:.5f}  {elapsed:.1f}s")
    print(f"  Test  raw={test_raw:.5f}  mae={test_mae:.6f} kW")
    print(f"  Best: {best.tree.to_lisp()[:100]}")

    return {
        "run_idx":      run_idx,
        "seed":         seed,
        "best":         best,
        "best_raw":     best.raw_fitness,
        "best_per_gen": best_per_gen,
        "success_gen":  success_gen,  # None if run hit MAX_GENERATIONS without succeeding
        "elapsed":      elapsed,
        "n_gen":        len(best_per_gen),
        "test_raw":     test_raw,
        "test_mae":     test_mae,
    }
