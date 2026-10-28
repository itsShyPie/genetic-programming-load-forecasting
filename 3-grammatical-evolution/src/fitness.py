#!/usr/bin/env python3
# fitness.py - fitness evaluation and tournament selection

import random
import numpy as np
from config import N_FITNESS_CASES, TOURNAMENT_SIZE
from primitives import eval_tree

# Penalty fitness for individuals whose derivation failed
_PENALTY = float(N_FITNESS_CASES * 10)


def evaluate_fitness(ind, X, Y):
    """
    Evaluate one individual and set its raw_fitness and adjusted_fitness.

    Invalid individuals (derivation failed) receive the maximum-penalty raw fitness
    so they are strongly disfavoured by selection.
    """
    if not ind.valid or ind.tree is None:
        ind.raw_fitness      = _PENALTY
        ind.adjusted_fitness = 1.0 / (1.0 + _PENALTY)
        return

    predictions = eval_tree(ind.tree, X)
    raw         = float(np.sum(np.abs(predictions - Y)))

    ind.raw_fitness      = raw
    ind.adjusted_fitness = 1.0 / (1.0 + raw)


def evaluate_population(pop, X, Y):
    """
    Evaluate every individual, then compute normalised fitness for roulette-wheel
    selection.
    """
    for ind in pop:
        evaluate_fitness(ind, X, Y)

    total = sum(ind.adjusted_fitness for ind in pop)
    if total < 1e-15:
        total = 1.0

    for ind in pop:
        ind.normalised_fitness = ind.adjusted_fitness / total


def tournament_select(pop):
    """
    k-Tournament selection: sample TOURNAMENT_SIZE individuals uniformly at random
    and return the one with the highest adjusted fitness.

    Tournament selection is preferred over fitness-proportionate selection for GE
    because it maintains selection pressure even when adjusted fitness values are
    clustered near zero (which happens when many invalid individuals are present).
    """
    contestants = random.sample(pop, min(TOURNAMENT_SIZE, len(pop)))
    return max(contestants, key=lambda ind: ind.adjusted_fitness)
