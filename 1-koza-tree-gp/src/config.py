#!/usr/bin/env python3
# config.py — every tunable value

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))

# DATASET #
DATA_PATH       = _os.path.join(_HERE, "..", "data", "Residential_Energy_Dataset_UK- 2014-2020.csv")

# Input: x_k = load at the same time-of-day, k days ago
#        (k * 96 steps back — 96 readings per day at 15-min intervals)
N_DAYS_BACK     = 7    # number of previous-day lags (x_1 … x_7)

TRAIN_RATIO     = 0.80  # first 80% of rows are used for training
N_FITNESS_CASES = 500   # how many training rows to randomly sample per run for fitness eval

# GP TREE #
POP_SIZE       = 500   # number of individuals in the population
MIN_INIT_DEPTH = 2     # minimum tree depth when generating the initial population
MAX_INIT_DEPTH = 8     # maximum tree depth when generating the initial population
MAX_TREE_DEPTH = 17    # hard cap on depth — offspring deeper than this are discarded
MAX_TREE_NODES = 100   # hard cap on node count — offspring larger than this are discarded

# Ephemeral Random Constants: literal float values baked into tree leaves at creation time
ERC_MIN = -5.0
ERC_MAX =  5.0

# OPERATOR RATES #
# These three must sum to 1.0
CROSSOVER_RATE    = 0.80  # swap subtrees between two parents
MUTATION_RATE     = 0.10  # replace a random subtree with a new random one
REPRODUCTION_RATE = 0.10  # copy an individual unchanged
assert abs(CROSSOVER_RATE + MUTATION_RATE + REPRODUCTION_RATE - 1.0) < 1e-9, \
    "Operator rates must sum to 1.0"

# RUN CONTROL #
MAX_GENERATIONS   = 250    # stop after this many generations even without success
SUCCESS_THRESHOLD = 0.14  # stop early if best adjusted_fitness >= this value

N_RUNS   = 20   # run the full GP this many times
BASE_SEED = 42  # run k uses seed = BASE_SEED + k, so every run is independently reproducible

# TERMINAL AND FUNCTION SETS
# Do not edit these directly

# How many input variables the GP will receive
N_INPUTS = N_DAYS_BACK

# Terminal set: input variables x_1..x_N, plus the ERC placeholder
# "ERC" is replaced by a random float when a tree node is created
TERMINAL_SET = [f"x_{i}" for i in range(1, N_INPUTS + 1)] + ["ERC"]

# Function set: maps each function name to its arity (number of arguments)
# div and sqrt are "protected" so they never produce NaN/Inf
FUNCTION_SET = {
    "add": 2, "sub": 2, "mul": 2, "div": 2,   # binary arithmetic
    "sqrt": 1, "sin": 1, "cos": 1,            # unary maths
    "neg": 1, "abs": 1,                       # unary sign/magnitude
}
FUNCTION_NAMES = list(FUNCTION_SET)  # plain list used when randomly picking a function
