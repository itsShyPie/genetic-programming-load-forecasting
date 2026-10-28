#!/usr/bin/env python3
# grammar.py - BNF grammar definition and GE derivation engine

from config import MAX_WRAPS, MAX_DEPTH, CODON_MAX
from tree import Node

GRAMMAR = {
    # Full numeric expression: arithmetic, unary, or a structural template
    "<expr>": [
        # Binary arithmetic - two recursive <expr> children
        ["add",  "<expr>", "<expr>"],   #  0
        ["sub",  "<expr>", "<expr>"],   #  1
        ["mul",  "<expr>", "<expr>"],   #  2
        ["div",  "<expr>", "<expr>"],   #  3
        # Unary functions - one recursive <expr> child
        ["neg",  "<expr>"],             #  4
        ["abs",  "<expr>"],             #  5
        ["sqrt", "<expr>"],             #  6
        # STRUCTURAL TEMPLATES - two <lexp> children (domain-specific building blocks)
        # These encode domain knowledge about electricity load patterns:
        ["lag_mean",  "<lexp>", "<lexp>"],   #  7  smoothed average
        ["lag_max",   "<lexp>", "<lexp>"],   #  8  peak load estimate
        ["lag_min",   "<lexp>", "<lexp>"],   #  9  base load estimate
        ["lag_delta", "<lexp>", "<lexp>"],   # 10  trend (day-over-day change)
        ["lag_ratio", "<lexp>", "<lexp>"],   # 11  relative change
    ],
    # "load expression" - produces a value in the load range.
    # First three rules allow a nested structural template (one level deep),
    # enabling expressions like lag_mean(lag_mean(x1, x7), lag_mean(x6, x1)).
    "<lexp>": [
        ["lag_mean", "<lag>", "<lag>"],  # 0  nested average of two raw lags
        ["lag_max",  "<lag>", "<lag>"],  # 1  nested max
        ["lag_min",  "<lag>", "<lag>"],  # 2  nested min
        ["<lag>"],                       # 3  single lag variable
    ],
    "<lag>": [
        ["x_1"], ["x_2"], ["x_3"], ["x_4"], ["x_5"], ["x_6"], ["x_7"],
    ],
}

# Indices in GRAMMAR["<expr>"] that do NOT recurse into <expr>.
# Used when the recursion depth limit is reached to force termination.
_SAFE_EXPR_INDICES = [7, 8, 9, 10, 11]   # the five structural templates


def _next_codon(codons, state):
    """
    Consume and return the next codon, wrapping if the end of the chromosome is
    reached.  Sets state["invalid"] = True if MAX_WRAPS is exceeded.
    """
    if state["pos"] >= len(codons):
        state["wraps"] += 1
        if state["wraps"] > MAX_WRAPS:
            state["invalid"] = True
            return 0
        state["pos"] = 0
    codon = codons[state["pos"]]
    state["pos"] += 1
    return codon


def derive(chromosome):
    """
    Derive an expression tree (Node) from an integer chromosome.

    Returns:
        (tree, valid)  where valid=False if derivation failed (too many wraps
        or depth limit encountered with no safe terminal available).
    """
    state = {"pos": 0, "wraps": 0, "invalid": False}
    tree  = _derive_expr(chromosome, state, MAX_DEPTH)
    valid = (not state["invalid"]) and (tree is not None)
    return tree, valid


def _derive_expr(codons, state, depth):
    if state["invalid"]:
        return None

    rules = GRAMMAR["<expr>"]

    if depth <= 0:
        # Depth limit: only non-recursive rules (structural templates) are allowed
        safe  = _SAFE_EXPR_INDICES
        codon = _next_codon(codons, state)
        if state["invalid"]:
            return None
        rule = rules[safe[codon % len(safe)]]
    else:
        codon = _next_codon(codons, state)
        if state["invalid"]:
            return None
        rule = rules[codon % len(rules)]

    head = rule[0]

    # Binary arithmetic
    if head in ("add", "sub", "mul", "div"):
        left  = _derive_expr(codons, state, depth - 1)
        right = _derive_expr(codons, state, depth - 1)
        if left is None or right is None:
            state["invalid"] = True
            return None
        return Node(head, [left, right])

    # Unary functions
    if head in ("neg", "abs", "sqrt"):
        child = _derive_expr(codons, state, depth - 1)
        if child is None:
            state["invalid"] = True
            return None
        return Node(head, [child])

    # Structural templates: two <lexp> children
    if head in ("lag_mean", "lag_max", "lag_min", "lag_delta", "lag_ratio"):
        lexp1 = _derive_lexp(codons, state)
        lexp2 = _derive_lexp(codons, state)
        if lexp1 is None or lexp2 is None:
            state["invalid"] = True
            return None
        return Node(head, [lexp1, lexp2])

    state["invalid"] = True
    return None


def _derive_lexp(codons, state):
    """Derive a <lexp> node: either a nested structural template or a single lag."""
    if state["invalid"]:
        return None

    lexp_rules = GRAMMAR["<lexp>"]
    codon      = _next_codon(codons, state)
    if state["invalid"]:
        return None

    rule = lexp_rules[codon % len(lexp_rules)]
    head = rule[0]

    # Nested structural template (one level deep): lag_mean/max/min of two raw lags
    if head in ("lag_mean", "lag_max", "lag_min"):
        lag1 = _derive_lag(codons, state)
        lag2 = _derive_lag(codons, state)
        if lag1 is None or lag2 is None:
            state["invalid"] = True
            return None
        return Node(head, [lag1, lag2])

    # Single lag variable
    if head == "<lag>":
        return _derive_lag(codons, state)

    state["invalid"] = True
    return None


def _derive_lag(codons, state):
    """Derive a <lag> terminal: one of x_1 .. x_7."""
    if state["invalid"]:
        return None
    lag_rules = GRAMMAR["<lag>"]
    codon     = _next_codon(codons, state)
    if state["invalid"]:
        return None
    name = lag_rules[codon % len(lag_rules)][0]   # e.g. "x_3"
    return Node(name)
