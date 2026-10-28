# data.py - loading the dataset and building the input/output arrays for fitness evaluation

import csv
import random
import numpy as np
from config import N_DAYS_BACK, TRAIN_RATIO


def load_dataset(path):
    """
    Read the CSV and return a plain list of electricity load values in time order.
    """
    loads = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                loads.append(float(row["Electricity_load"]))
            except (ValueError, KeyError):
                pass   # skip malformed rows
    return loads


def build_fitness_cases(loads, n_samples, rng):
    """
    Build the training examples the GP uses to measure fitness.

    Returns:
        X - dict {"x_1": array, "x_2": array, ...}, each array length n_samples
        Y - array of actual loads at each sampled time t, length n_samples
    """
    train_end = int(len(loads) * TRAIN_RATIO)

    # offset = how far back furthest lag reaches; can't use rows before this index
    offset = N_DAYS_BACK * 96

    # Randomly pick row indices within the valid training range
    sampled = rng.sample(range(offset, train_end), min(n_samples, train_end - offset))

    # For each lag k, go back k full days (k * 96 steps)
    X = {
        f"x_{k}": np.array([loads[i - k * 96] for i in sampled], dtype=np.float64)
        for k in range(1, N_DAYS_BACK + 1)
    }

    # Target: the actual load at each sampled time t
    Y = np.array([loads[i] for i in sampled], dtype=np.float64)

    return X, Y


def build_test_set(loads):
    """
    Return the full held-out test set as (X, Y) arrays - no sampling.
    Uses every row from train_end onward that has enough lag history.
    """
    train_end = int(len(loads) * TRAIN_RATIO)
    offset    = N_DAYS_BACK * 96
    indices   = range(max(train_end, offset), len(loads))
    X = {
        f"x_{k}": np.array([loads[i - k * 96] for i in indices], dtype=np.float64)
        for k in range(1, N_DAYS_BACK + 1)
    }
    Y = np.array([loads[i] for i in indices], dtype=np.float64)
    return X, Y
