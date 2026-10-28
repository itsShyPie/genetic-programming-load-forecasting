# population.py - building the initial population of GP trees

import random
from config import (
    ERC_MIN, ERC_MAX,
    FUNCTION_NAMES, FUNCTION_SET,
    MIN_INIT_DEPTH, MAX_INIT_DEPTH,
    TERMINAL_SET,
)
from tree import Individual, Node


def _terminal():
    """
    Pick a random terminal node.
    If we land on "ERC", generate a random float and store it directly in the node
    (that float is then permanently part of this tree).
    """
    t = random.choice(TERMINAL_SET)
    if t == "ERC":
        return Node(round(random.uniform(ERC_MIN, ERC_MAX), 4))  # e.g. Node(2.3817)
    return Node(t)   # e.g. Node("x_3")


def _full(d):
    """
    Full method: every internal node is a function; leaves are forced at exactly depth d.
    """
    if d == 0:
        return _terminal()   # we've reached the target depth - place a leaf
    f = random.choice(FUNCTION_NAMES)
    # recursively build one child branch per argument the function needs
    return Node(f, [_full(d - 1) for _ in range(FUNCTION_SET[f])])


def _grow(d):
    """
    Grow method: at each level, randomly choose a function OR a terminal.
    """
    if d == 0:
        return _terminal()   # hit the depth limit - must place a leaf
    choice = random.choice(FUNCTION_NAMES + TERMINAL_SET)
    if choice in FUNCTION_SET:
        # it's a function - keep building downward
        return Node(choice, [_grow(d - 1) for _ in range(FUNCTION_SET[choice])])
    # it's a terminal - stop here even if we haven't hit d yet
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
    """
    depths = list(range(MIN_INIT_DEPTH, MAX_INIT_DEPTH + 1))
    seen   = set()    # set of expression strings we've already added
    pop    = []

    # Try up to pop_size*100 times - duplicates are discarded, so we need extra attempts
    for _ in range(pop_size * 100):
        if len(pop) >= pop_size:
            break

        idx = len(pop)

        # Cycle through depths so each depth gets an equal share
        d = depths[idx % len(depths)]

        # Within each depth level, alternate Full and Grow to get the 50/50 split
        tree = _full(d) if (idx // len(depths)) % 2 == 0 else _grow(d)

        # Only add if this exact expression hasn't been seen yet
        expr = tree.to_expr()
        if expr not in seen:
            seen.add(expr)
            pop.append(Individual(tree))

    return pop
