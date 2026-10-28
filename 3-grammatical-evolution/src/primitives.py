#!/usr/bin/env python3
# primitives.py - protected math operations and vectorised tree evaluation

import numpy as np

_CLIP = 1e6


def _pdiv(a, b):
    safe = np.where(np.abs(b) < 1e-10, 1.0, b)
    return np.clip(np.where(np.abs(b) < 1e-10, 1.0, a / safe), -_CLIP, _CLIP)


def _psqrt(a):
    return np.clip(np.sqrt(np.abs(a)), 0.0, _CLIP)


def _safe(arr):
    return np.clip(np.nan_to_num(arr, nan=_CLIP, posinf=_CLIP, neginf=-_CLIP), -_CLIP, _CLIP)


def _lag_mean(a, b):  return (a + b) * 0.5
def _lag_max(a, b):   return np.maximum(a, b)
def _lag_min(a, b):   return np.minimum(a, b)
def _lag_delta(a, b): return a - b
def _lag_ratio(a, b): return _pdiv(a, b)


def eval_tree(node, X):
    """
    Evaluate a Node tree over a batch of inputs.
    X is a dict {"x_1": array, ..., "x_7": array}, each of shape (n,).
    Returns a 1-D numpy array of n predictions.
    """
    n = len(next(iter(X.values())))

    if node.is_terminal:
        if isinstance(node.value, float):
            return np.full(n, node.value, dtype=np.float64)
        return X[node.value].copy()

    v    = node.value
    args = [eval_tree(c, X) for c in node.children]

    with np.errstate(all="ignore"):
        if   v == "add":       res = args[0] + args[1]
        elif v == "sub":       res = args[0] - args[1]
        elif v == "mul":       res = args[0] * args[1]
        elif v == "div":       res = _pdiv(args[0], args[1])
        elif v == "sqrt":      res = _psqrt(args[0])
        elif v == "sin":       res = np.sin(np.clip(args[0], -1e6, 1e6))
        elif v == "cos":       res = np.cos(np.clip(args[0], -1e6, 1e6))
        elif v == "neg":       res = -args[0]
        elif v == "abs":       res = np.abs(args[0])
        elif v == "lag_mean":  res = _lag_mean(args[0], args[1])
        elif v == "lag_max":   res = _lag_max(args[0], args[1])
        elif v == "lag_min":   res = _lag_min(args[0], args[1])
        elif v == "lag_delta": res = _lag_delta(args[0], args[1])
        elif v == "lag_ratio": res = _lag_ratio(args[0], args[1])
        else:
            raise ValueError(f"Unknown primitive: {v!r}")

    return _safe(res)
