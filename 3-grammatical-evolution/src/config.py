#!/usr/bin/env python3
# config.py - all tunable parameters for the Structure-Based GE run

import os
_HERE = os.path.dirname(os.path.abspath(__file__))

# DATASET
DATA_PATH = os.path.join(
    _HERE, "..", "data", "Residential_Energy_Dataset_UK- 2014-2020.csv"
)

# INPUT FEATURES
# x_k = electricity load at the same time-of-day, k days ago (k * 96 steps back)
N_DAYS_BACK = 7   # x_1 .. x_7

# TRAIN / TEST SPLIT
TRAIN_RATIO     = 0.80
# Larger sample than A2 (500) to address feedback
N_FITNESS_CASES = 1500

# GE CHROMOSOME
# Kept short (30 codons) so that crossover frequently hits active codons rather
# than introns.  A typical derivation uses 5-15 codons; CHROM_LEN=30 keeps the
# intron fraction around 50%, making single-point crossover meaningful.
CHROM_LEN  = 30    # number of integer codons per individual
CODON_MAX  = 255   # each codon drawn uniformly from [0, CODON_MAX]
MAX_WRAPS  = 3     # max times the chromosome can wrap during derivation
MAX_DEPTH  = 8     # max recursion depth when deriving the expression tree

# POPULATION
POP_SIZE = 500

# OPERATOR RATES - must sum exactly to 1.0
CROSSOVER_RATE    = 0.80
MUTATION_RATE     = 0.15
REPRODUCTION_RATE = 0.05
assert abs(CROSSOVER_RATE + MUTATION_RATE + REPRODUCTION_RATE - 1.0) < 1e-9, \
    "Operator rates must sum to 1.0"

# Per-codon probability of random reset during mutation.
# Expected mutations per chromosome: 30 * 0.05 = 1.5
CODON_MUTATION_PROB = 0.05

# SELECTION
# Smaller tournament (k=4) reduces selection pressure, preserving more diversity
# than the k=7 used in the first run (which caused immediate premature convergence).
TOURNAMENT_SIZE = 4   # k-tournament: pick best of k random contestants

# RUN CONTROL
# No early-exit success threshold - run all generations to allow proper convergence.
MAX_GENERATIONS = 200
N_RUNS          = 10
BASE_SEED       = 42  # run k uses seed BASE_SEED + k
