# operators.py - the three genetic operators that create new individuals

import random
from config import MAX_INIT_DEPTH, MAX_TREE_DEPTH, MAX_TREE_NODES
from population import _grow   # reuse the Grow generator for mutation's new subtree


# TREE NAVIGATION HELPERS #
# Crossover and mutation both need to pick a random node and then swap/replace it.
# Nodes numbered in DFS order (root = 0, depth-first left-to-right)
# to reach any node by a single integer index.

def _count(node):
    # Total number of nodes in this subtree
    return 1 + sum(_count(c) for c in node.children)


def _get(root, idx):
    """
    Return (node, parent, child_index) for the node at DFS position idx.
    """
    counter = [0]    # mutable counter shared across recursive calls
    result  = [None]

    def dfs(node, parent, ci):
        if result[0]:            # already found - stop recursing
            return
        if counter[0] == idx:
            result[0] = (node, parent, ci)
            return
        counter[0] += 1
        for i, child in enumerate(node.children):
            dfs(child, node, i)

    dfs(root, None, None)
    return result[0] or (root, None, None)   # fallback to root if idx somehow out of range


# OPERATORS #

def reproduce(ind):
    """
    Reproduction: copy the individual unchanged into the next generation.
    No modification at all - preserves fit individuals across generations.
    """
    return ind.copy()


def crossover(p1, p2):
    """
    Subtree crossover: take a subtree from p2 and graft it onto p1.

    Step by step:
        1. Deep-copy p1 so we don't modify the original.
        2. Pick a random non-root node in the copy (the "cut point").
        3. Pick a random node anywhere in p2 (the "donor subtree").
        4. Replace the cut-point node in the copy with the donor subtree.
        5. If the result is too deep or too large, fall back to a copy of p1.
    """
    child = p1.copy()
    n1    = _count(child.tree)
    n2    = _count(p2.tree)

    # Pick cut point in child (prefer non-root so we actually mix the trees)
    pt1 = random.randint(1, n1 - 1) if n1 > 1 else 0
    # Pick donor subtree root anywhere in p2
    pt2 = random.randint(0, n2 - 1)

    _, par1, ci1 = _get(child.tree, pt1)   # where to cut in child
    donor, _, _  = _get(p2.tree, pt2)      # what to graft in

    donated = donor.copy()   # copy the donor so p2 is not modified

    if par1:
        par1.children[ci1] = donated  # replace the subtree at the cut point
    else:
        child.tree = donated          # cut point was root — replace the whole tree

    # Reject offspring that violate size/depth limits - return parent copy instead
    # This ensures no bloated trees enter the population
    if child.depth() > MAX_TREE_DEPTH or child.size() > MAX_TREE_NODES:
        return p1.copy()
    return child


def mutate(ind):
    """
    Subtree mutation: replace a random subtree with a brand-new random one.

    Step by step:
        1. Deep-copy the individual.
        2. Pick a random node in the copy (the mutation point).
        3. Generate a fresh random subtree using the Grow method.
        4. Replace the node at the mutation point with the new subtree.
        5. If the result is too deep or too large, return the original unchanged.
    """
    child = ind.copy()
    n     = _count(child.tree)
    pt    = random.randint(0, n - 1)   # any node, including root

    _, parent, ci = _get(child.tree, pt)

    # New random subtree - depth between 1 and MAX_INIT_DEPTH
    new_subtree = _grow(random.randint(1, MAX_INIT_DEPTH))

    if parent:
        parent.children[ci] = new_subtree  # replace at the chosen point
    else:
        child.tree = new_subtree            # mutation point was root

    if child.depth() > MAX_TREE_DEPTH or child.size() > MAX_TREE_NODES:
        return ind.copy()
    return child
