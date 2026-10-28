# stgp_types.py - STGP type system for electricity load forecasting

# Two types reflect the domain structure:
#   REAL: any real-valued expression (ERC constants, function outputs)
#   LOAD: an electricity load reading (terminal variables x_1..x_7, always
#         in [0.031, 0.177] kW and strictly positive)
#
# Subtyping rule: LOAD satisfies REAL (a load value is a real number), but
# REAL does NOT satisfy LOAD (an arbitrary real is not necessarily a valid
# load reading).  This is enforced during type-aware tree generation and the
# typed crossover/mutation operators.

REAL = "REAL"
LOAD = "LOAD"

# FUNCTION SIGNATURES
# Each entry: function_name -> {"args": [arg_type, ...], "ret": return_type}

FUNCTION_SIGNATURES = {
    # Standard arithmetic and maths (all REAL -> REAL)
    "add":  {"args": [REAL, REAL], "ret": REAL},
    "sub":  {"args": [REAL, REAL], "ret": REAL},
    "mul":  {"args": [REAL, REAL], "ret": REAL},
    "div":  {"args": [REAL, REAL], "ret": REAL},
    "sqrt": {"args": [REAL],       "ret": REAL},
    "sin":  {"args": [REAL],       "ret": REAL},
    "cos":  {"args": [REAL],       "ret": REAL},
    "neg":  {"args": [REAL],       "ret": REAL},
    "abs":  {"args": [REAL],       "ret": REAL},

    # Domain-typed functions (LOAD x LOAD -> LOAD)
    # These preserve the LOAD type - their output is still a valid load value.
    "lag_mean": {"args": [LOAD, LOAD], "ret": LOAD},  # arithmetic mean of two loads
    "lag_max":  {"args": [LOAD, LOAD], "ret": LOAD},  # maximum of two loads
    "lag_min":  {"args": [LOAD, LOAD], "ret": LOAD},  # minimum of two loads

    # Domain-typed functions (LOAD x LOAD -> REAL)
    # These return a real (not necessarily a valid load value).
    "lag_delta": {"args": [LOAD, LOAD], "ret": REAL},  # signed difference a - b
    "lag_ratio": {"args": [LOAD, LOAD], "ret": REAL},  # protected ratio a / b
}

# Arity of every function (derived automatically)
FUNCTION_ARITY = {name: len(sig["args"]) for name, sig in FUNCTION_SIGNATURES.items()}

# All function names as a list
FUNCTION_NAMES = list(FUNCTION_SIGNATURES)

# Partitioned by return type - used when selecting a function for a given return type
FUNCTIONS_RETURNING = {
    REAL: [n for n, s in FUNCTION_SIGNATURES.items() if s["ret"] == REAL],
    LOAD: [n for n, s in FUNCTION_SIGNATURES.items() if s["ret"] == LOAD],
}

# TERMINAL TYPES

def terminal_type(value):
    """
    Return the STGP type of a terminal node value.

    float values are ERCs (Ephemeral Random Constants) - they are arbitrary
    real numbers, so their type is REAL.
    String values starting with "x_" are lag variables - they hold actual
    electricity load readings, so their type is LOAD.
    """
    if isinstance(value, float):
        return REAL   # ERC - arbitrary constant
    if isinstance(value, str) and value.startswith("x_"):
        return LOAD   # lag variable - an actual electricity load reading
    raise ValueError(f"Unknown terminal value: {value!r}")


# SUBTYPING

def satisfies(provided_type, required_type):
    """
    Return True if a node of provided_type may be used where required_type is needed.

    Rules:
        Every type satisfies itself.
        LOAD satisfies REAL (LOAD is a subtype of REAL).
        REAL does NOT satisfy LOAD.
    """
    if provided_type == required_type:
        return True
    if provided_type == LOAD and required_type == REAL:
        return True
    return False
