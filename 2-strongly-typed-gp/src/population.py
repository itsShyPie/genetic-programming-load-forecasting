# population.py - building the initial population of STGP trees

import random

from config import (
    ERC_MIN, ERC_MAX, MIN_INIT_DEPTH, MAX_INIT_DEPTH,
    MAX_TREE_NODES, TERMINAL_SET,
)
from stgp_types import (
    REAL, LOAD,
    FUNCTIONS_RETURNING, FUNCTION_SIGNATURES, FUNCTION_ARITY,
    terminal_type, satisfies,
)
from tree import Individual, Node


def _make_terminal(required_type):
    """
    Pick a random terminal Node whose type satisfies required_type.

    If required_type is LOAD only lag variables are eligible (ERC is a REAL,
    not a valid load reading, so it cannot satisfy LOAD).
    If required_type is REAL both lag variables and ERCs are eligible since
    LOAD satisfies REAL.
    """
    if required_type == LOAD:
        candidates = [t for t in TERMINAL_SET if t != "ERC"]
    else:  # REAL - any terminal works (LOAD satisfies REAL)
        candidates = TERMINAL_SET

    t = random.choice(candidates)
    if t == "ERC":
        return Node(round(random.uniform(ERC_MIN, ERC_MAX), 4))  # e.g. Node(2.3817)
    return Node(t)   # e.g. Node("x_3")


def _full(d, required_type=REAL):
    """
    Full method: every path terminates at exactly depth d.
    All nodes above the leaf level are function nodes.
    required_type is the return type the subtree root must satisfy.
    """
    if d == 0:
        return _make_terminal(required_type)   # reached the target depth - place a leaf

    # Collect function names whose return type satisfies required_type
    candidates = [
        name
        for name, sig in FUNCTION_SIGNATURES.items()
        if satisfies(sig["ret"], required_type)
    ]
    if not candidates:
        return _make_terminal(required_type)

    fname = random.choice(candidates)
    arg_types = FUNCTION_SIGNATURES[fname]["args"]
    # recursively build one child branch per argument function needs
    children  = [_full(d - 1, at) for at in arg_types]
    return Node(fname, children)


def _grow(d, required_type=REAL):
    """
    Grow method: at each level, randomly choose a function OR a terminal
    that satisfies required_type.  Terminals can appear before depth d.
    """
    if d == 0:
        return _make_terminal(required_type)   # hit the depth limit - must place a leaf

    # Functions whose return satisfies required_type
    func_candidates = [
        name
        for name, sig in FUNCTION_SIGNATURES.items()
        if satisfies(sig["ret"], required_type)
    ]

    # Terminals that satisfy required_type
    if required_type == LOAD:
        term_candidates = [t for t in TERMINAL_SET if t != "ERC"]
    else:
        term_candidates = TERMINAL_SET

    all_candidates = func_candidates + term_candidates
    if not all_candidates:
        return _make_terminal(required_type)

    choice = random.choice(all_candidates)

    if choice in FUNCTION_SIGNATURES:
        # it's a function - keep building downward
        arg_types = FUNCTION_SIGNATURES[choice]["args"]
        children  = [_grow(d - 1, at) for at in arg_types]
        return Node(choice, children)

    # it's a terminal - stop here even if haven't hit d yet
    if choice == "ERC":
        return Node(round(random.uniform(ERC_MIN, ERC_MAX), 4))
    return Node(choice)


def ramped_half_and_half(pop_size):
    """
    Ramped Half-and-Half: the recommended GP initialisation method (Koza 1992).

    This method creates variety by:
        1. Splitting the population equally across depth levels MIN..MAX.
        2. Using Full for half the slots at each depth, Grow for the other half.
        3. Rejecting duplicates (via expression-string hashing) until every
           individual in the population is unique - 100% variety guaranteed.
        4. Rejecting trees that exceed MAX_TREE_NODES (bushy LOAD-typed subtrees
           can be very large at depth 8 with arity-2 functions).

    All generated trees have root type REAL (the output must be a number).
    """
    depths = list(range(MIN_INIT_DEPTH, MAX_INIT_DEPTH + 1))
    seen   = set()    # set of expression strings we've already added
    pop    = []

    # Try up to pop_size*200 times - duplicates and oversized trees are discarded
    for _ in range(pop_size * 200):
        if len(pop) >= pop_size:
            break

        idx = len(pop)

        # Cycle through depths each depth gets an equal share
        d = depths[idx % len(depths)]

        # Within each depth level, alternate Full and Grow to get the 50/50 split
        tree = _full(d, REAL) if (idx // len(depths)) % 2 == 0 else _grow(d, REAL)

        # Reject trees that exceed the node-count cap (can happen with arity-2
        # LOAD functions producing bushy subtrees at depth 8)
        if tree.size() > MAX_TREE_NODES:
            continue

        # Only add if this exact expression hasn't been seen yet
        expr = tree.to_expr()
        if expr not in seen:
            seen.add(expr)
            pop.append(Individual(tree))

    return pop
