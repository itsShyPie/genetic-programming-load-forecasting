# tree.py - the two core data structures used throughout the GP

from stgp_types import terminal_type, FUNCTION_SIGNATURES


class Node:
    """
    A single node in a STGP parse tree.

    Two kinds:
        Function node - value is a string like "add" or "lag_mean", children is a non-empty list
        Terminal node - value is a variable name like "x_1" OR a float constant; children is []

    Every node carries a node_type attribute ("REAL" or "LOAD") set once at
    construction time.  This is what makes the tree strongly-typed: crossover
    and mutation use node_type to ensure only type-compatible subtrees are swapped.
    """
    # __slots__ speeds up attribute access and reduces memory use per node
    __slots__ = ("value", "children", "node_type")

    def __init__(self, value, children=None, node_type=None):
        self.value    = value
        self.children = children or []
        # Infer type if not provided explicitly
        if node_type is not None:
            self.node_type = node_type
        elif self.children:
            # function node - look up return type from the signature table
            self.node_type = FUNCTION_SIGNATURES[value]["ret"]
        else:
            # terminal node - type depends on whether it's an ERC or a lag variable
            self.node_type = terminal_type(value)

    # TREE SHAPE #

    @property
    def is_terminal(self):
        # a node is a leaf when it has no children
        return not self.children

    def depth(self):
        # depth of root = 0; each function layer adds 1
        return 0 if self.is_terminal else 1 + max(c.depth() for c in self.children)

    def size(self):
        # total node count in this subtree (including self)
        return 1 + sum(c.size() for c in self.children)

    # STRING REPRESENTATIONS

    def to_expr(self):
        # Returns a canonical string like "(x_1+psqrt(x_2))"
        # Used only for duplicate detection during population init - not for eval
        if self.is_terminal:
            # floats are ERC values; other terminals are variable names
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
        # Human readable prefix notation: (add x_1 (lag_mean x_6 x_7))
        # This is what gets printed when showing the best individual
        if self.is_terminal:
            return f"{self.value:.4f}" if isinstance(self.value, float) else str(self.value)
        return f"({self.value} {' '.join(c.to_lisp() for c in self.children)})"

    def copy(self):
        # deep copy - needed before applying genetic operators so we don't mutate the original
        return Node(self.value, [c.copy() for c in self.children], node_type=self.node_type)

    def __repr__(self):
        return self.to_lisp()


class Individual:
    """
    A GP individual = one parse tree + its four Koza (1992) fitness scores.

    The four fitness values form a chain (all set by evaluate_fitness in fitness.py):

        raw_fitness = sum|predicted - actual| over all fitness cases lower is better; 0
        would be a perfect fit

        standardised_fitness = same as raw here (raw is already >= 0, 0 = perfect)

        adjusted_fitness = 1 / (1 + standardised) maps to (0, 1] where 1 = perfect this
        is the value we compare against SUCCESS_THRESHOLD

        normalised_fitness = adjusted_i / sum(adjusted for whole population) each
        individual's share of the total; sums to 1.0 used as the probability of
        being selected (roulette wheel)
    """

    def __init__(self, tree):
        self.tree                 = tree
        self.raw_fitness          = None
        self.standardised_fitness = None
        self.adjusted_fitness     = None
        self.normalised_fitness   = None

    def depth(self): return self.tree.depth()
    def size(self):  return self.tree.size()
    def copy(self):  return Individual(self.tree.copy())
