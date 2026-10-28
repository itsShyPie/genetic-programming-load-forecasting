# operators.py - the three genetic operators that create new individuals

import random

from config import MAX_INIT_DEPTH, MAX_TREE_DEPTH, MAX_TREE_NODES
from stgp_types import satisfies, REAL
from population import _grow   # reuse the Grow generator for mutation's new subtree


# TREE NAVIGATION HELPERS
# Crossover and mutation both need to pick a random node and then swap/replace it.
# Nodes are numbered in DFS order (root = 0, depth-first left-to-right)
# so any node can be reached by a single integer index.

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


def _collect_indices_of_type(root, target_type):
    """
    Return a list of all DFS indices whose node_type satisfies target_type.
    Used to find type-compatible swap points in STGP crossover.
    """
    result  = []
    counter = [0]

    def dfs(node):
        if satisfies(node.node_type, target_type):
            result.append(counter[0])
        counter[0] += 1
        for child in node.children:
            dfs(child)

    dfs(root)
    return result


# OPERATORS #

def reproduce(ind):
    """
    Reproduction: copy the individual unchanged into the next generation.
    No modification at all - preserves fit individuals across generations.
    """
    return ind.copy()


def crossover(p1, p2):
    """
    Type-safe subtree crossover (STGP crossover).

    Step by step:
        1. Deep-copy p1 so we don't modify the original.
        2. Pick a random non-root node in the copy (the "cut point") with type T.
        3. Find all nodes in p2 whose return type satisfies T (compatible donors).
        4. If no compatible donor exists, return a copy of p1 unchanged.
        5. Graft the selected donor subtree at the cut point.
        6. If the result is too deep or too large, fall back to a copy of p1.

    Type safety guarantee: the resulting tree is always well-typed because
    we only substitute subtrees of compatible return types.
    """
    child = p1.copy()
    n1    = _count(child.tree)

    # Pick cut point in child (prefer non-root so actually mix the trees)
    pt1 = random.randint(1, n1 - 1) if n1 > 1 else 0
    cut_node, par1, ci1 = _get(child.tree, pt1)
    required_type = cut_node.node_type   # type that the donor must satisfy

    # Find compatible nodes in p2
    compatible_indices = _collect_indices_of_type(p2.tree, required_type)
    if not compatible_indices:
        return p1.copy()   # no type-compatible donor - fallback to copy

    pt2 = random.choice(compatible_indices)
    donor, _, _ = _get(p2.tree, pt2)
    donated = donor.copy()   # copy the donor so p2 is not modified

    if par1:
        par1.children[ci1] = donated  # replace the subtree at the cut point
    else:
        child.tree = donated          # cut point was root - replace the whole tree

    # Reject offspring that violate size/depth limits - return parent copy instead
    # This ensures no bloated trees enter the population
    if child.depth() > MAX_TREE_DEPTH or child.size() > MAX_TREE_NODES:
        return p1.copy()
    return child


def mutate(ind):
    """
    Type-safe subtree mutation.

    Step by step:
        1. Deep-copy the individual.
        2. Pick a random node in the copy (the mutation point); its type is T.
        3. Generate a fresh random subtree of return type T using the Grow method.
        4. Replace the node at the mutation point with the new subtree.
        5. If the result is too deep or too large, return the original unchanged.

    Type safety guarantee: the replacement subtree has the same return type
    as the node it replaces, so the tree remains well-typed.
    """
    child = ind.copy()
    n     = _count(child.tree)
    pt    = random.randint(0, n - 1)   # any node, including root

    target_node, parent, ci = _get(child.tree, pt)
    target_type = target_node.node_type

    # New random subtree of required type - depth between 1 and MAX_INIT_DEPTH
    new_subtree = _grow(random.randint(1, MAX_INIT_DEPTH), target_type)

    if parent:
        parent.children[ci] = new_subtree  # replace at the chosen point
    else:
        child.tree = new_subtree            # mutation point was root

    if child.depth() > MAX_TREE_DEPTH or child.size() > MAX_TREE_NODES:
        return ind.copy()
    return child
