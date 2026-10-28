#!/usr/bin/env python3
# operators.py - GE genetic operators: crossover, mutation, reproduction

import random
from config import CHROM_LEN, CODON_MAX, CODON_MUTATION_PROB
from population import make_individual_from_chromosome


def reproduce(ind):
    """Return an exact copy of the individual (no modification)."""
    return ind.copy()


def crossover(p1, p2):
    """
    Single-point crossover on the integer chromosomes.

    1. Pick a random cut point in [1, CHROM_LEN - 1].
    2. Offspring chromosome = p1[:cut] + p2[cut:]
    3. Re-derive the expression tree from the new chromosome.

    Single-point crossover preserves the codon ordering at the head of the
    dominant parent (p1), potentially inheriting the derivation prefix.
    """
    cut    = random.randint(1, CHROM_LEN - 1)
    chrom  = p1.chromosome[:cut] + p2.chromosome[cut:]
    return make_individual_from_chromosome(chrom)


def mutate(ind):
    """
    Per-codon random-reset mutation.

    Each codon is independently replaced by a new random value in [0, CODON_MAX]
    with probability CODON_MUTATION_PROB.  The tree is re-derived afterward.

    If no codon is mutated (all flips missed), the individual is returned
    unchanged (same as reproduction).  This is correct behaviour — with
    CHROM_LEN=128 and rate=0.01 the expected number of mutations is 1.28,
    so most offspring receive at least one change.
    """
    chrom = ind.chromosome[:]
    for i in range(CHROM_LEN):
        if random.random() < CODON_MUTATION_PROB:
            chrom[i] = random.randint(0, CODON_MAX)
    return make_individual_from_chromosome(chrom)
