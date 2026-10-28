#!/usr/bin/env python3
# tree.py - Node and Individual data structures for GE

class Node:
    """
    A single node in a derived expression tree.

    Function node: value is a string like "add" or "lag_mean", children is non-empty.
    Terminal node: value is a variable name ("x_1") or a float ERC; children is [].
    """
    __slots__ = ("value", "children")

    def __init__(self, value, children=None):
        self.value    = value
        self.children = children or []

    @property
    def is_terminal(self):
        return not self.children

    def depth(self):
        return 0 if self.is_terminal else 1 + max(c.depth() for c in self.children)

    def size(self):
        return 1 + sum(c.size() for c in self.children)

    def to_expr(self):
        """Canonical string used for duplicate detection."""
        if self.is_terminal:
            return f"{self.value:.6g}" if isinstance(self.value, float) else str(self.value)
        v, a = self.value, [c.to_expr() for c in self.children]
        if   v == "add":       return f"({a[0]}+{a[1]})"
        elif v == "sub":       return f"({a[0]}-{a[1]})"
        elif v == "mul":       return f"({a[0]}*{a[1]})"
        elif v == "div":       return f"pdiv({a[0]},{a[1]})"
        elif v == "sqrt":      return f"psqrt({a[0]})"
        elif v == "sin":       return f"sin({a[0]})"
        elif v == "cos":       return f"cos({a[0]})"
        elif v == "neg":       return f"(-{a[0]})"
        elif v == "abs":       return f"abs({a[0]})"
        elif v == "lag_mean":  return f"lag_mean({a[0]},{a[1]})"
        elif v == "lag_max":   return f"lag_max({a[0]},{a[1]})"
        elif v == "lag_min":   return f"lag_min({a[0]},{a[1]})"
        elif v == "lag_delta": return f"lag_delta({a[0]},{a[1]})"
        elif v == "lag_ratio": return f"lag_ratio({a[0]},{a[1]})"
        else:                  return f"{v}({','.join(a)})"

    def to_lisp(self):
        """Human-readable prefix notation for display."""
        if self.is_terminal:
            return f"{self.value:.4f}" if isinstance(self.value, float) else str(self.value)
        return f"({self.value} {' '.join(c.to_lisp() for c in self.children)})"

    def copy(self):
        return Node(self.value, [c.copy() for c in self.children])

    def __repr__(self):
        return self.to_lisp()


class Individual:
    """
    A GE individual = integer chromosome + derived expression tree + fitness scores.

    The genotype (chromosome) is a list of integers in [0, CODON_MAX].
    The phenotype (tree) is derived by grammar.derive(chromosome).
    If derivation fails (too many wraps, or grammar can't terminate), valid=False
    and the individual receives maximum-penalty fitness.
    """

    def __init__(self, chromosome, tree=None, valid=True):
        self.chromosome       = chromosome[:]   # list[int], length CHROM_LEN
        self.tree             = tree            # Node | None
        self.valid            = valid           # False if derivation failed

        self.raw_fitness      = None
        self.adjusted_fitness = None
        self.normalised_fitness = None

    def depth(self): return self.tree.depth() if self.tree else 0
    def size(self):  return self.tree.size()  if self.tree else 0

    def copy(self):
        ind = Individual(
            self.chromosome,
            self.tree.copy() if self.tree else None,
            self.valid,
        )
        ind.raw_fitness       = self.raw_fitness
        ind.adjusted_fitness  = self.adjusted_fitness
        ind.normalised_fitness = self.normalised_fitness
        return ind
