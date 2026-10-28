#!/usr/bin/env python3
# data.py - dataset loading and fitness-case construction

import csv
import random
import numpy as np
from config import N_DAYS_BACK, TRAIN_RATIO


def load_dataset(path):
    """Read the CSV and return a plain list of electricity load floats in time order."""
    loads = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                loads.append(float(row["Electricity_load"]))
            except (ValueError, KeyError):
                pass
    return loads


def build_fitness_cases(loads, n_samples, rng):
    """
    Sample n_samples training rows and return (X, Y).

    X — dict {"x_1": array, ..., "x_7": array}, each of shape (n_samples,)
    Y — target load array of shape (n_samples,)

    Each lag x_k is the load at the same time-of-day k days earlier (k * 96 steps back).
    """
    train_end = int(len(loads) * TRAIN_RATIO)
    offset    = N_DAYS_BACK * 96   # earliest index with full lag history

    sampled = rng.sample(
        range(offset, train_end),
        min(n_samples, train_end - offset),
    )

    X = {
        f"x_{k}": np.array([loads[i - k * 96] for i in sampled], dtype=np.float64)
        for k in range(1, N_DAYS_BACK + 1)
    }
    Y = np.array([loads[i] for i in sampled], dtype=np.float64)
    return X, Y


def build_test_set(loads):
    """Return the full held-out test set (no sampling) as (X, Y)."""
    train_end = int(len(loads) * TRAIN_RATIO)
    offset    = N_DAYS_BACK * 96
    indices   = range(max(train_end, offset), len(loads))
    X = {
        f"x_{k}": np.array([loads[i - k * 96] for i in indices], dtype=np.float64)
        for k in range(1, N_DAYS_BACK + 1)
    }
    Y = np.array([loads[i] for i in indices], dtype=np.float64)
    return X, Y
