# fitness.py - evaluating how good each individual is, and selecting parents

import random
import numpy as np
from primitives import eval_tree

def evaluate_fitness(ind, X, Y):
    """
    Evaluate one individual and fill in all four fitness values.

    X - dict of input arrays (one per lag variable), shape (n_cases,)
    Y - target array of actual electricity loads, shape (n_cases,)
    """
    # Run the tree over all fitness cases at once (vectorised via numpy)
    predictions = eval_tree(ind.tree, X)

    # Raw fitness = sum of absolute errors across all fitness cases
    raw = float(np.sum(np.abs(predictions - Y)))

    ind.raw_fitness          = raw
    ind.standardised_fitness = raw                 # same as raw: already >= 0, 0 = perfect
    ind.adjusted_fitness     = 1.0 / (1.0 + raw)   # flip so higher = better; range (0, 1]


def evaluate_population(pop, X, Y):
    """
    Evaluate every individual in the population, then compute normalised fitness.

    After this call:
        Every individual has all four fitness values set.
        All normalised_fitness values sum to 1.0 (ready for roulette-wheel selection).

    This is called once per generation, as required by the core GP loop.
    """
    for ind in pop:
        evaluate_fitness(ind, X, Y)

    # Sum all adjusted values so we can normalise them
    total = sum(ind.adjusted_fitness for ind in pop)
    if total < 1e-15:
        total = 1.0   # fallback: give everyone an equal share if all are terrible

    for ind in pop:
        # Each individual's share of the total — used directly as selection probability
        ind.normalised_fitness = ind.adjusted_fitness / total


# SELECTION #

def fps(pop):
    """
    Fitness-Proportionate Selection (roulette wheel).

    Draw a random number r in [0, 1].
    Walk through the population accumulating normalised_fitness values.
    Return the first individual whose cumulative total reaches r.
    """
    r, cumulative = random.random(), 0.0
    for ind in pop:
        cumulative += ind.normalised_fitness
        if r <= cumulative:
            return ind
    return pop[-1]   # safety fallback for floating-point rounding
