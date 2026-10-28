# primitives.py - protected math operations and vectorised tree evaluation

import numpy as np

# Clip result to +/-_CLIP to stop overflow cascading through deep trees.
_CLIP = 1e6


def _pdiv(a, b):
    # Protected division: return 1.0 wherever the denominator is effectively zero,
    # otherwise compute a/b and clip to +/-_CLIP
    safe = np.where(np.abs(b) < 1e-10, 1.0, b)          # avoid dividing by ~0
    return np.clip(np.where(np.abs(b) < 1e-10, 1.0, a / safe), -_CLIP, _CLIP)


def _psqrt(a):
    # Protected square root: take sqrt(|a|) negative inputs are safe
    return np.clip(np.sqrt(np.abs(a)), 0.0, _CLIP)


def _safe(arr):
    # Final safety after every operation:
    # replace NaN and +/-Inf with _CLIP, clamp everything to +/-_CLIP
    return np.clip(np.nan_to_num(arr, nan=_CLIP, posinf=_CLIP, neginf=-_CLIP), -_CLIP, _CLIP)


# DOMAIN-TYPED FUNCTIONS
# These operate on LOAD values (electricity load readings).
# They extend A1's function set to give the GP domain-aware building blocks.

def _lag_mean(a, b):
    # Arithmetic mean of two load values - output stays in the load range
    return (a + b) * 0.5


def _lag_max(a, b):
    # Element-wise maximum of two load arrays
    return np.maximum(a, b)


def _lag_min(a, b):
    # Element-wise minimum of two load arrays
    return np.minimum(a, b)


def _lag_delta(a, b):
    # Signed difference a - b. Captures week-over-week change in load.
    return a - b


def _lag_ratio(a, b):
    # Protected ratio a / b. Captures relative change between two lag days.
    return _pdiv(a, b)


# VECTORIZED TREE EVALUATION

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
        return X[node.value].copy()                          # variable: its array of values

    # Function node: recursively evaluate children first, apply the operation
    v    = node.value
    args = [eval_tree(c, X) for c in node.children]   # evaluate left and/or right subtree

    with np.errstate(all="ignore"):   # silence numpy warnings
        if   v == "add":       res = args[0] + args[1]
        elif v == "sub":       res = args[0] - args[1]
        elif v == "mul":       res = args[0] * args[1]
        elif v == "div":       res = _pdiv(args[0], args[1])        # protected - no div-by-zero
        elif v == "sqrt":      res = _psqrt(args[0])                # protected - no sqrt(-ve)
        elif v == "sin":       res = np.sin(np.clip(args[0], -1e6, 1e6))   # clip before trig
        elif v == "cos":       res = np.cos(np.clip(args[0], -1e6, 1e6))   # to avoid domain error
        elif v == "neg":       res = -args[0]
        elif v == "abs":       res = np.abs(args[0])
        # Domain-typed primitives
        elif v == "lag_mean":  res = _lag_mean(args[0], args[1])
        elif v == "lag_max":   res = _lag_max(args[0], args[1])
        elif v == "lag_min":   res = _lag_min(args[0], args[1])
        elif v == "lag_delta": res = _lag_delta(args[0], args[1])
        elif v == "lag_ratio": res = _lag_ratio(args[0], args[1])
        else:
            raise ValueError(f"Unknown primitive: {v!r}")

    return _safe(res)   # clean up any remaining NaN/Inf before returning
