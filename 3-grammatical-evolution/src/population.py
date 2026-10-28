#!/usr/bin/env python3
# population.py - initial population generation for GE

import random
from config import CHROM_LEN, CODON_MAX, POP_SIZE
from grammar import derive
from tree import Individual


def _random_chromosome():
    """Generate one random integer chromosome of length CHROM_LEN."""
    return [random.randint(0, CODON_MAX) for _ in range(CHROM_LEN)]


def _make_individual():
    """Create one Individual: random chromosome -> derive tree -> wrap in Individual."""
    chrom      = _random_chromosome()
    tree, valid = derive(chrom)
    return Individual(chrom, tree, valid)


def random_population(pop_size=POP_SIZE):
    """
    Generate an initial population of pop_size GE individuals.

    Each individual has a randomly generated integer chromosome.  The grammar
    derivation is run immediately so the tree (phenotype) is available from the
    start.  Invalid individuals (derivation failed) are kept - they receive
    penalty fitness during evaluation and are unlikely to survive selection.

    No uniqueness filtering is applied; GE chromosomes are long enough that
    random collisions in phenotype space are extremely rare.
    """
    return [_make_individual() for _ in range(pop_size)]


def make_individual_from_chromosome(chrom):
    """Derive and wrap a pre-built chromosome into an Individual."""
    tree, valid = derive(chrom)
    return Individual(chrom, tree, valid)
