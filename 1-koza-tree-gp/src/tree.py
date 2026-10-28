# tree.py - the two core data structures used throughout the GP


class Node:
    """
    A single node in a GP parse tree.
    
    Two kinds:
        Function node — value is a string like "add", children is a non-empty list
        Terminal node — value is a variable name like "x_1" OR a float constant; children is []
    """
    # __slots__ speeds up attribute access and reduces memory use per node
    __slots__ = ("value", "children")

    def __init__(self, value, children=None):
        self.value    = value
        self.children = children or []

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

    # STRING REPRESENTATIONS #

    def to_expr(self):
        # Returns a canonical string like "(x_1+psqrt(x_2))"
        # Used only for duplicate detection during population init - not for eval
        if self.is_terminal:
            # floats are ERC values; other terminals are variable names
            return f"{self.value:.6g}" if isinstance(self.value, float) else str(self.value)
        v, a = self.value, [c.to_expr() for c in self.children]
        if   v == "add":  return f"({a[0]}+{a[1]})"
        elif v == "sub":  return f"({a[0]}-{a[1]})"
        elif v == "mul":  return f"({a[0]}*{a[1]})"
        elif v == "div":  return f"pdiv({a[0]},{a[1]})"
        elif v == "sqrt": return f"psqrt({a[0]})"
        elif v == "sin":  return f"sin({a[0]})"
        elif v == "cos":  return f"cos({a[0]})"
        elif v == "neg":  return f"(-{a[0]})"
        else:             return f"abs({a[0]})"

    def to_lisp(self):
        # Human readable prefix notation: (add x_1 (mul x_2 3.0))
        # This is what gets printed when showing the best individual
        if self.is_terminal:
            return f"{self.value:.4f}" if isinstance(self.value, float) else str(self.value)
        return f"({self.value} {' '.join(c.to_lisp() for c in self.children)})"

    def copy(self):
        # deep copy - needed before applying genetic operators so we don't mutate the original
        return Node(self.value, [c.copy() for c in self.children])

    def __repr__(self):
        return self.to_lisp()


class Individual:
    """
    A GP individual = one parse tree + its four Koza (1992) fitness scores.

    The four fitness values form a chain (all set by evaluate_fitness in fitness.py):
        
        raw_fitness = Σ|predicted - actual| over all fitness cases lower is better; 0
        would be a perfect fit
        
        standardised_fitness = same as raw here (raw is already >= 0, 0 = perfect)
        
        adjusted_fitness = 1 / (1 + standardised) maps to (0, 1] where 1 = perfect this
        is the value we compare against SUCCESS_THRESHOLD
        
        normalised_fitness = adjusted_i / Σ(adjusted for whole population) each
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
