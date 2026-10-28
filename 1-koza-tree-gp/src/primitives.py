# primitives.py - protected math operations and vectorised tree evaluation

import numpy as np
from tree import Node

# Clip every result to ±_CLIP to stop overflow cascading through deep trees.
_CLIP = 1e6


def _pdiv(a, b):
    # Protected division: return 1.0 wherever the denominator is effectively zero,
    # otherwise compute a/b and clip to ±_CLIP
    safe = np.where(np.abs(b) < 1e-10, 1.0, b)          # avoid dividing by ~0
    return np.clip(np.where(np.abs(b) < 1e-10, 1.0, a / safe), -_CLIP, _CLIP)


def _psqrt(a):
    # Protected square root: take sqrt(|a|) so negative inputs are safe
    return np.clip(np.sqrt(np.abs(a)), 0.0, _CLIP)


def _safe(arr):
    # Final safety after every operation:
    # replace NaN and ±Inf with _CLIP, then clamp everything to ±_CLIP
    return np.clip(np.nan_to_num(arr, nan=_CLIP, posinf=_CLIP, neginf=-_CLIP), -_CLIP, _CLIP)


# VECTORIZED TREE EVALUATION #

def eval_tree(node, X):
    """
    Walk the tree and evaluate it over a whole batch of inputs at once.

    Instead of looping over 500 fitness cases one at a time, we pass numpy
    arrays through the tree. Each node returns a 1-D array of 500 values.
    This is faster than a Python loop.
    """
    n = len(next(iter(X.values())))   # number of fitness cases

    # Terminal node: return a constant array or look up the variable's values
    if node.is_terminal:
        if isinstance(node.value, float):
            return np.full(n, node.value, dtype=np.float64)  # ERC: same constant for every case
        return X[node.value].copy()                           # variable: its array of values

    # Function node: recursively evaluate children first, then apply the operation
    v    = node.value
    args = [eval_tree(c, X) for c in node.children]   # evaluate left and/or right subtree

    with np.errstate(all="ignore"):   # silence numpy warnings - protected ops handle them
        if   v == "add":  res = args[0] + args[1]
        elif v == "sub":  res = args[0] - args[1]
        elif v == "mul":  res = args[0] * args[1]
        elif v == "div":  res = _pdiv(args[0], args[1])       # protected — no div-by-zero
        elif v == "sqrt": res = _psqrt(args[0])               # protected — no sqrt(-ve)
        elif v == "sin":  res = np.sin(np.clip(args[0], -1e6, 1e6))   # clip before trig
        elif v == "cos":  res = np.cos(np.clip(args[0], -1e6, 1e6))   #  to avoid domain error
        elif v == "neg":  res = -args[0]
        elif v == "abs":  res = np.abs(args[0])

    return _safe(res)   # clean up any remaining NaN/Inf before returning
