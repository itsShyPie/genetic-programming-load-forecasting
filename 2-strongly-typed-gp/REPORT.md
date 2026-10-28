# Structure-Based Genetic Programming for Electricity Load Forecasting
---

## 1. Description of the Dataset

### Source and Overview

The dataset used in this project is the UK Residential Energy Dataset (2014-2020), a time-series dataset of residential electricity consumption recorded at 15-minute intervals. The data is stored in a CSV file and contains 201,604 readings spanning from 31 December 2014 to 30 September 2020 which is a period of approximately 5.75 years (2,100 days). The dataset provides a comprehensive view of residential energy behavior across multiple seasons and years, making it good for learning and evaluating time-series forecasting models.

### Dataset Characteristics

| Property            | Value                                     |
| ------------------- | ----------------------------------------- |
| Total rows          | 201,604                                   |
| Temporal resolution | 15-minute intervals (96 readings per day) |
| Date range          | 31/12/2014 - 30/09/2020                   |
| Target variable     | `Electricity_load` (kW per household)     |
| Target minimum      | 0.031 kW                                  |
| Target maximum      | 0.177 kW                                  |
| Target range        | 0.146 kW                                  |
| Target mean         | 0.0946 kW                                 |
| Target std dev      | 0.0227 kW                                 |
| Target median       | 0.093 kW                                  |
| Missing values      | None                                      |

The target variable is tightly bounded (minimum 0.031 kW, maximum 0.177 kW), which is consistent with the relatively uniform electricity consumption patterns of a single residential household. The low standard deviation (0.0227 kW, approximately 24% of the mean) indicates moderate variability that is driven primarily by time-of-day and day-of-week cycles rather than extreme outliers.

### Columns

|Column|Description|Used|
|---|---|---|
|`utc_timestamp`|UTC date and time of the reading|No|
|`Electricity_load`|Residential electricity consumption (kW)|**Yes**|
|`Residential_electricity_price`|Electricity price (GBP)|No|
|`Residential_solar_generation`|Solar energy generation (kW)|No|
|`Residential_wind_generation`|Wind energy generation (kW)|No|
|`Temperature`|Ambient air temperature (°C)|No|
|`Relative Humidity`|Relative humidity (%)|No|

Only the `Electricity_load` column is used because this is treated as a time-series problem: can load at time _t_ be predicted from lagged load values at earlier times, without relying on external covariates? Using only lagged load values keeps the search space clean and ensures that results are attributable to the GP alone.

### Data Preprocessing

No preprocessing was required as the dataset contains no missing values across all 201,604 rows. No normalization or scaling was applied to the target variable, as the GP operates on raw load values and the fitness measure (sum of absolute errors) is expressed in the same units (kW), making the threshold directly readable.

### Train/Test Split

The dataset is divided chronologically into a training set (first 80%) and a held-out test set (remaining 20%):

| Partition | Rows    | Date Range                          |
| --------- | ------- | ----------------------------------- |
| Training  | 161,283 | 31/12/2014 - 07/08/2019 23:30       |
| Test      | 40,321  | 07/08/2019 23:45 - 30/09/2020 23:45 |

The split is chronological rather than random. Randomly splitting temporal data would leak future information into training, whereas a chronological split correctly simulates real deployment conditions: train on the past, evaluate on the unseen future.

### Input Variable Construction

Seven lagged input features are constructed, each representing the electricity load at the same time-of-day on each of the previous seven days. Since there are 96 readings per day, the lag for day _k_ is _k_ × 96 time steps back:

> x₁ = load at t - 96, x₂ = load at t - 192, x₃ = load at t - 288, x₄ = load at t - 384, x₅ = load at t - 480, x₆ = load at t - 576, x₇ = load at t - 672

This design reflects the periodic structure of residential electricity consumption at both the 24-hour scale (sleep cycles, cooking, heating) and the 7-day scale (weekday vs. weekend patterns). By anchoring each lag to the same time-of-day, the input features create directly comparable historical snapshots.

### Fitness Cases

Because evaluating all 161,283 training rows each generation would be computationally expensive, a random sample of 500 rows is drawn from the training set once at the start of each GP run and fixed for all generations within that run. This keeps the fitness signal consistent: if the sample changed each generation, an individual's fitness could change for reasons unrelated to its true quality, introducing noise into the selection process.

---

## 2. Performance Metric

### Fitness Measure: The Koza (1992) Four-Step Chain

This implementation adopts the complete four-level fitness transformation as described by Koza (1992), progressing from a raw error measure to a normalised probability suitable for selection. The chain is implemented in `fitness.py`.

| Step         | Formula                     | Direction                     |
| ------------ | --------------------------- | ----------------------------- |
| Raw          | `Σ\|ŷ - y\|` over 500 cases | lower = better; 0 = perfect   |
| Standardised | = raw (already ≥ 0)         | lower = better                |
| Adjusted     | `1 / (1 + raw)`             | higher = better, ∈ (0, 1]     |
| Normalised   | `adjusted_i / Σ adjusted`   | sums to 1.0 across population |

#### Step 1 - Raw Fitness

Raw fitness is defined as the sum of absolute errors across all 500 fitness cases:

$$r(i) = \sum_{j=1}^{N} \left| \hat{y}_j - y_j \right|$$

where $\hat{y}_j$ is the GP tree's prediction for fitness case $j$ and $y_j$ is the actual electricity load. A raw fitness of 0 represents a perfect predictor. Lower values are better.

Why sum of absolute errors? SAE is less sensitive to outlier fitness cases than sum of squared errors and produces values in the same units as the target variable (kW), making thresholds directly interpretable. The success threshold of `SUCCESS_THRESHOLD = 0.14` corresponds to a raw fitness of $1/0.14 - 1 \approx 6.143$, which equals a mean absolute error of $6.143 / 500 \approx 0.01229$ kW per case (approximately 8.42% of the full target range of 0.146 kW).

#### Step 2 - Standardized Fitness

Since sum of absolute errors already satisfies the property that lower is better and 0 is optimal, standardized fitness is set equal to raw fitness:

$$s(i) = r(i)$$

#### Step 3 - Adjusted Fitness

Adjusted fitness maps standardized fitness to the range **(0, 1]**, where higher is better and 1 represents a perfect individual:

$$a(i) = \frac{1}{1 + s(i)}$$

This transformation compresses the unbounded raw fitness onto a bounded scale. It is particularly important for fitness-proportionate selection because it ensures that even very poor individuals receive a small but non-zero selection probability, preventing complete starvation of weaker individuals early in the run.

#### Step 4 - Normalized Fitness

Normalized fitness expresses each individual's adjusted fitness as a fraction of the total adjusted fitness across the entire population:

$$n(i) = \frac{a(i)}{\sum_{j=1}^{M} a(j)}$$

All normalized fitness values across a population of size $M$ sum to exactly 1.0, making them directly usable as selection probabilities for roulette-wheel selection.

### Edge Case Handling

Numerical edge cases are handled in `primitives.py`:

- Division by zero: Protected division (`pdiv`) returns 1.0 when the denominator's absolute value falls below 1×10⁻¹⁰, and clips the result to ±10⁶ otherwise.
- Square root of negative numbers: Protected square root (`psqrt`) computes `sqrt(|a|)`, ensuring a real-valued result for any input.
- Overflow and NaN propagation: A final safety pass (`_safe`) replaces any remaining NaN or ±Inf values with ±10⁶ and clips all outputs to the range $[−10⁶, +10⁶]$.
- Zero total adjusted fitness: If all individuals produce degenerate outputs and the total adjusted fitness falls below 1×10⁻¹⁵, the normalization denominator is set to 1.0, giving each individual an equal selection probability.

---

## 3. Structure-Based Genetic Programming

### What is Structure-Based GP?

Structure-based GP, also called Strongly Typed GP (STGP), extends canonical GP by assigning a data type to every node in the parse tree - both function nodes (whose return type is determined by their definition) and terminal nodes (whose type is fixed at creation). The type system is enforced throughout the entire GP lifecycle: tree initialisation, crossover, and mutation all respect type constraints, so every tree in the population is guaranteed to be well-typed at all times.

In canonical GP, any subtree can be swapped with any other subtree during crossover, regardless of the types of the values they produce. This can lead to semantically meaningless expressions - for example, using the result of a trigonometric function as if it were a load reading. STGP prevents this by requiring that crossover can only substitute a subtree of the same (or compatible) return type. This is the structural constraint that distinguishes A2 from A1.

### The Type System

The type system is defined in `stgp_types.py` and consists of two types:

| Type   | Meaning                                                                                   |
| ------ | ----------------------------------------------------------------------------------------- |
| `REAL` | Any real-valued expression - ERC constants, arithmetic function outputs                   |
| `LOAD` | An electricity load reading - terminal variables x₁ … x₇, strictly in \[0.031, 0.177\] kW |

The key design decision is that LOAD is a subtype of REAL: a load value is a real number, so a LOAD node may be used wherever a REAL is expected. However, a REAL node may not be used where a LOAD is expected, because an arbitrary real number is not necessarily a valid load reading. This asymmetric subtyping relationship is captured by the `satisfies(provided, required)` function:

```python
def satisfies(provided_type, required_type):
    if provided_type == required_type: return True
    if provided_type == LOAD and required_type == REAL: return True
    return False
```

This means `satisfies(LOAD, REAL) = True` but `satisfies(REAL, LOAD) = False`.

### Why REAL and LOAD Rather Than Boolean?

One could instead use a Boolean/Integer distinction, but that distinction is only semantically meaningful for problems that involve conditional logic (if-then-else, while loops) where Boolean values appear naturally. Electricity load forecasting is a pure regression problem: there are no Boolean values anywhere in the domain. The REAL/LOAD distinction is more meaningful for this problem because it separates two genuinely different classes of value - arbitrary scalars versus bounded, positive, physically-interpretable load readings - and allows domain-typed functions to be defined that are only valid on load values.

### Type Assignment

Every node in the tree carries a `node_type` attribute set once at construction time and never changed:

- **Function nodes:** Return type is looked up from `FUNCTION_SIGNATURES[name]["ret"]`. For example, `lag_mean` returns `LOAD`; `add` returns `REAL`.
- **Terminal nodes:** ERC float constants have type `REAL`; lag variables x₁ … x₇ have type `LOAD`.

### Type-Safe Tree Generation

When building a tree that must satisfy a required type at its root, the generator only selects functions whose return type satisfies the required type. For `_full(d, required_type=REAL)`:

```python
candidates = [name for name, sig in FUNCTION_SIGNATURES.items()
              if satisfies(sig["ret"], required_type)]
```

For terminals, if `LOAD` is required, ERC is excluded because a random float is not a valid load reading. If `REAL` is required, both lag variables and ERCs are eligible since `satisfies(LOAD, REAL)` is True.

### Type-Safe Crossover

In type-safe STGP crossover, the cut point's type constrains which donor subtrees are eligible:

1. A random non-root node is selected in parent 1 as the cut point; its type is recorded as `required_type`.
2. All nodes in parent 2 whose return type satisfies `required_type` are collected as compatible donors.
3. If no compatible donor exists, a copy of parent 1 is returned unchanged.
4. A donor is chosen at random from the compatible set and grafted at the cut point.

This guarantees that after crossover, the offspring's tree is still well-typed: the subtree inserted at the cut point has the same return type as the subtree it replaced, so no type constraint anywhere in the tree is violated.

### Type-Safe Mutation

Subtree mutation selects a random mutation point and records its type `T`. A new subtree is generated using the type-aware Grow method with required type `T`. The new subtree is guaranteed to have return type `T`, so replacing the old subtree does not break any type constraint in the tree.

---

## 4. Representation: Terminal and Function Sets

### Terminal Set

The terminal set consists of two categories of leaf node, each with an assigned STGP type:

| Terminal        | STGP Type | Description                                                                     |
| --------------- | --------- | ------------------------------------------------------------------------------- |
| x₁, x₂, …, x₇ | `LOAD`    | Lagged electricity load values (one per previous day at the same time-of-day)  |
| ERC             | `REAL`    | Ephemeral Random Constant: a float drawn uniformly from [−5.0, 5.0] at creation |

Why are lag variables typed as LOAD? Each xₖ holds an actual electricity load reading from the dataset, bounded in \[0.031, 0.177\] kW and strictly positive. Assigning type LOAD to these variables makes it possible to define domain-typed functions that operate specifically on load readings - combining two load values by taking their mean, max, or min produces a result that is still a load reading, which is physically meaningful. By contrast, adding an arbitrary ERC to a load reading may produce a value outside the valid load range, which is why ERCs are typed REAL and cannot be passed to LOAD-typed function arguments.

### Function Set

The function set is extended from A1 with five domain-typed functions that exploit the LOAD type:

#### Standard Functions (REAL × REAL → REAL)

| Function | Arity | Description                                                   |
| -------- | ----- | ------------------------------------------------------------- |
| `add`    | 2     | Addition                                                      |
| `sub`    | 2     | Subtraction                                                   |
| `mul`    | 2     | Multiplication                                                |
| `div`    | 2     | **Protected** division - returns 1.0 when denominator ≈ 0     |
| `sqrt`   | 1     | **Protected** square root - computes √\|a\|                   |
| `sin`    | 1     | Sine (argument clipped to \[-10⁶, +10⁶\] before evaluation)   |
| `cos`    | 1     | Cosine (argument clipped to \[-10⁶, +10⁶\] before evaluation) |
| `neg`    | 1     | Unary negation                                                |
| `abs`    | 1     | Absolute value                                                |

#### Domain-Typed Functions (LOAD × LOAD → LOAD or REAL)

| Function    | Signature          | Description                                                        |
| ----------- | ------------------ | ------------------------------------------------------------------ |
| `lag_mean`  | LOAD × LOAD → LOAD | Arithmetic mean of two load values; output is a valid load reading |
| `lag_max`   | LOAD × LOAD → LOAD | Element-wise maximum of two load arrays                            |
| `lag_min`   | LOAD × LOAD → LOAD | Element-wise minimum of two load arrays                            |
| `lag_delta` | LOAD × LOAD → REAL | Signed difference a - b; captures week-over-week change in load    |
| `lag_ratio` | LOAD × LOAD → REAL | Protected ratio a / b; captures relative change between lag days   |

Why these domain-typed functions? The five new functions encode physically meaningful operations on load readings:
- `lag_mean`, `lag_max`, `lag_min` produce a load-range output from two load inputs, allowing the GP to discover expressions like "the average of yesterday's and last week's load" without needing to construct that structure from raw arithmetic.
- `lag_delta` and `lag_ratio` produce real-valued summaries of the change between two load readings, allowing the GP to detect trends and relative variations.

Without the STGP type system these functions could not be safely defined - their arguments must be bounded positive values, and canonical GP would have no way to enforce this. The type system guarantees that `lag_mean(x₁, x₂)` always receives valid load readings as both arguments, and that its output (still a load reading) can be fed into another LOAD-typed function such as `lag_max`.

### Closure Property

Every function accepts inputs compatible with its declared argument types and returns a value of its declared return type. Protected operations (`div`, `sqrt`, `lag_ratio`) guarantee this: `div` and `lag_ratio` handle the denominator-zero case explicitly; `sqrt` takes the absolute value of its argument before computing the root. The combination of type constraints and protected operations ensures that every tree in the population always evaluates to a finite, real-valued number regardless of inputs.

### Sufficiency Property

The combination of the nine standard arithmetic/trigonometric functions, five domain-typed functions, seven lagged input variables, and ERCs is sufficient to approximate a wide class of time-series patterns, including periodic, trend-following, and autoregressive behavior. The domain-typed functions additionally allow the GP to construct weighted and comparative summaries of lagged loads directly, without having to rediscover these structures from raw arithmetic.

---

## 5. Initial Population Generation

### Method: Type-Aware Ramped Half-and-Half

The initial population is generated using Ramped Half-and-Half (Koza, 1992), extended to respect the STGP type system. The method is implemented in `population.py`.

#### Depth and Method Cycling

1. Tree depths cycle through `MIN_INIT_DEPTH = 2` to `MAX_INIT_DEPTH = 8` (seven depth levels). With 500 individuals across 7 levels, depths are distributed near-uniformly.
2. Within each depth, generation alternates between Full and Grow, producing approximately half Full and half Grow trees at each depth.

#### Type-Aware Terminal Selection

The `_make_terminal(required_type)` function respects the type requirement:
- If `LOAD` is required, only lag variables x₁ … x₇ are eligible (ERC is excluded because a random float is not a valid load reading).
- If `REAL` is required, all terminals are eligible since `satisfies(LOAD, REAL)` is True.

#### Type-Aware Full Method

`_full(d, required_type)` builds a tree of exact depth `d` whose root satisfies `required_type`. At each non-leaf level, only functions whose return type satisfies `required_type` are considered as candidates. Each function's argument types are then recursively passed down, so the entire tree is constructed consistently from the root's type requirement.

#### Type-Aware Grow Method

`_grow(d, required_type)` builds a tree of at most depth `d`. At each level it chooses randomly from functions returning a compatible type or terminals of a compatible type. This produces structurally varied trees of smaller expected size than the Full method.

#### Duplicate Rejection and Size Capping

Each generated tree is checked against all previously accepted trees via its canonical expression string. Trees that have already been added are discarded, guaranteeing that every slot in the initial population is structurally distinct. Additionally, trees that exceed `MAX_TREE_NODES = 100` are discarded: the LOAD-typed functions are all arity-2, so Full trees at depth 8 built entirely from LOAD functions can contain up to 255 nodes, far exceeding the cap. A retry budget of `pop_size × 200` ensures the target population size is reached despite these rejections.

---

## 6. Fitness Function and Fitness Evaluation

### The `evaluate_fitness` Function

The fitness of a single individual is computed by `evaluate_fitness(ind, X, Y)` in `fitness.py`. This function:

1. Vectorised tree evaluation: Passes the individual's parse tree and the full dictionary of input arrays `X` to `eval_tree()` in `primitives.py`, which evaluates the tree recursively using NumPy array operations. Each node operation computes all 500 fitness cases in a single array operation, which is faster than evaluating one case at a time in Python.

2. Raw fitness: Computes fitness as the sum of absolute differences between the tree's predictions and the actual load values in `Y`.

3. Fitness storage: Stores the raw, standardized, and adjusted fitness values on the individual object. Normalized fitness is computed separately once all individuals in the population have been evaluated.

### Population-Level Evaluation

`evaluate_population(pop, X, Y)` is called once per generation. It iterates over every individual in the population, calling `evaluate_fitness` for each, and then computes normalised fitness values as each individual's adjusted fitness divided by the population total.

### Fitness Cases and Consistency

The same 500 fitness cases are used for every generation within a run. If cases were re-sampled each generation, an individual might appear to improve or regress solely because the sample changed. Fixed cases make fitness a stable, reproducible property of the individual.

---

## 7. Selection Method

### Method: Fitness-Proportionate Selection (Roulette Wheel)

Selection is implemented by the `fps(pop)` function in `fitness.py` using the standard roulette-wheel method. Fitness-Proportionate Selection (FPS) is the natural pairing for the normalized fitness measure: normalized fitness values sum to 1.0 and directly form a valid probability distribution.

Under FPS, an individual with twice the normalized fitness is twice as likely to be selected. This gives proportional selection pressure that favors fitter individuals while still preserving a non-zero survival chance for weaker ones.

The algorithm proceeds as follows:

1. Draw a random number _r_ uniformly from $[0, 1)$.
2. Walk through the population accumulating each individual's normalized fitness into a running total.
3. Return the first individual whose cumulative total reaches or exceeds _r_.
4. If floating-point rounding causes no individual to be selected, return the last individual as a safety fallback.

---

## 8. Genetic Operators

All three operators are implemented in `operators.py` using the STGP-extended versions that enforce type safety.

| Operator     | Rate | Function                  |
| ------------ | ---- | ------------------------- |
| Crossover    | 0.80 | `crossover(p1, p2)`       |
| Mutation     | 0.10 | `mutate(ind)`             |
| Reproduction | 0.10 | `reproduce(ind)`          |

### Correct Operator Application - Key Fix vs A1

In Assignment 1, operators were applied by drawing a random number per offspring and comparing it against cumulative rates. This is stochastic: the actual number of offspring produced by each operator in any given generation was left to chance and could deviate from the intended rates.

In A2, **exact counts** are computed from the rates before filling the new population:

```python
n_crossover    = round(POP_SIZE * CROSSOVER_RATE)       # exactly 400
n_mutation     = round(POP_SIZE * MUTATION_RATE)        # exactly  50
n_reproduction = POP_SIZE - n_crossover - n_mutation    # exactly  50
```

Exactly 400 individuals are produced by crossover, exactly 50 by mutation, and exactly 50 by reproduction in every generation, without any random deviation.

### 8.1 Reproduction - `reproduce(ind)`

The individual is deep-copied and placed unchanged into the next generation. Reproduction preserves high-quality individuals across generations without risking the destruction of good structures.

### 8.2 Type-Safe Subtree Crossover - `crossover(p1, p2)`

Type-safe crossover ensures that the offspring's tree remains well-typed by enforcing type compatibility between the cut point and the donor subtree:

1. Deep-copy parent 1 to form the offspring.
2. Select a random non-root cut point in the offspring; record its return type `T`.
3. Collect all nodes in parent 2 whose return type satisfies `T` - these are the compatible donors.
4. If no compatible donor exists, return a copy of parent 1 unchanged.
5. Select a donor at random from the compatible set and graft a copy of it at the cut point.
6. If the resulting tree exceeds `MAX_TREE_DEPTH = 17` or `MAX_TREE_NODES = 100`, return a copy of parent 1 instead.

Type safety guarantee: The subtree inserted at the cut point has the same return type as the subtree it replaced. Since the rest of the tree is unchanged, every type constraint in the offspring is satisfied.

Why type-safe crossover matters here: Consider a `lag_mean` node that requires two `LOAD` arguments. In canonical GP, crossover might replace one of those arguments with an ERC or the output of `sin`, both of which are `REAL` type and not valid load readings. Type-safe crossover prevents this: only subtrees of type `LOAD` can be inserted as children of `lag_mean`, so the resulting tree is always semantically valid.

### 8.3 Type-Safe Subtree Mutation - `mutate(ind)`

Type-safe mutation ensures that the replacement subtree has the same return type as the node it replaces:

1. Deep-copy the individual.
2. Select a random mutation point anywhere in the tree; record its return type `T`.
3. Generate a fresh random subtree of return type `T` using the type-aware Grow method.
4. Replace the subtree at the mutation point with the new subtree.
5. If the result exceeds the depth or node count limits, return the original individual unchanged.

Type safety guarantee: The replacement subtree has return type `T`, matching the node it replaces, so the tree remains well-typed after mutation.

### Size and Depth Enforcement

Both crossover and mutation enforce hard limits of `MAX_TREE_DEPTH = 17` and `MAX_TREE_NODES = 100`. These caps prevent the GP bloat phenomenon - the tendency for trees to grow without bound due to crossover's statistical bias toward selecting larger subtrees - by discarding oversized offspring and returning the parent unchanged.

---

## 9. Experimental Setup

### Control Model

A generational control model with single-individual elitism is used. The generational model replaces the entire population each generation. Elitism ensures that the best individual from the current generation is always copied into the next generation unchanged, guaranteeing that the best-ever fitness trajectory is monotonically non-increasing in raw error.

### Termination Criterion

A run terminates when either of the following conditions is met:

1. The maximum number of generations (`MAX_GENERATIONS = 250`) is reached.
2. The best individual's adjusted fitness reaches or exceeds `SUCCESS_THRESHOLD = 0.14`, corresponding to a mean absolute error of at most 0.01229 kW per fitness case (8.42% of the full target range).

### Complete Parameter Configuration

All parameters are centralized in `config.py`.

| Parameter           | Value | Description                                    |
| ------------------- | ----- | ---------------------------------------------- |
| `N_DAYS_BACK`       | 7     | Number of daily lag inputs (x₁ … x₇)           |
| `TRAIN_RATIO`       | 0.80  | First 80% of data used for training            |
| `N_FITNESS_CASES`   | 500   | Training samples per run                       |
| `POP_SIZE`          | 500   | Individuals per generation                     |
| `MIN_INIT_DEPTH`    | 2     | Minimum tree depth at initialisation           |
| `MAX_INIT_DEPTH`    | 8     | Maximum tree depth at initialisation           |
| `MAX_TREE_DEPTH`    | 17    | Hard depth cap on offspring                    |
| `MAX_TREE_NODES`    | 100   | Hard node-count cap on offspring               |
| `ERC_MIN`           | -5.0  | Lower bound for ephemeral constants            |
| `ERC_MAX`           | +5.0  | Upper bound for ephemeral constants            |
| `CROSSOVER_RATE`    | 0.80  | Fraction of offspring produced by crossover    |
| `MUTATION_RATE`     | 0.10  | Fraction of offspring produced by mutation     |
| `REPRODUCTION_RATE` | 0.10  | Fraction of offspring produced by reproduction |
| `MAX_GENERATIONS`   | 250   | Maximum generations per run                    |
| `SUCCESS_THRESHOLD` | 0.14  | Early-stop adjusted fitness threshold          |
| `N_RUNS`            | 20    | Number of independent runs                     |
| `BASE_SEED`         | 42    | Run _k_ uses seed BASE_SEED + _k_              |

### Reproducibility

Each run uses a fixed random seed of `BASE_SEED + run_index` applied to both Python's built-in `random` module and NumPy's random generator. This guarantees that every run can be reproduced identically by re-running the code with the same configuration.

### Technical Specifications

| Component          | Value                                                |
| ------------------ | ---------------------------------------------------- |
| Python             | 3.12.3                                               |
| NumPy              | 1.26.4                                               |
| OS                 | Linux WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2) |
| External libraries | NumPy only                                           |
| GP implementation  | From scratch                                         |

Twenty independent runs are performed with different random seeds (seeds 42-61). A single GP run is stochastic, so twenty runs provide enough samples to compute meaningful statistics and to apply the Koza (1992) computational effort metric.

---

## 10. Results and Discussion

### Per-Run Summary

Twenty independent runs were performed (seeds 42-61, up to 250 generations each, `POP_SIZE = 500`, `N_FITNESS_CASES = 500`):

| Run | Seed | Success Gen | Final Adj | Final Raw | Time (s) | Test MAE (kW) |
| --- | ---- | ----------- | --------- | --------- | -------- | ------------- |
| 1   | 42   | **0**       | 0.14348   | 5.96975   | 0.9      | 0.017935      |
| 2   | 43   | -           | 0.13829   | 6.23119   | 67.2     | 0.018169      |
| 3   | 44   | -           | 0.13415   | 6.45425   | 59.0     | 0.018551      |
| 4   | 45   | **48**      | 0.14013   | 6.13619   | 9.2      | 0.017904      |
| 5   | 46   | -           | 0.13482   | 6.41703   | 64.9     | 0.017894      |
| 6   | 47   | -           | 0.13703   | 6.29753   | 77.3     | 0.017802      |
| 7   | 48   | -           | 0.13448   | 6.43607   | 69.2     | 0.018125      |
| 8   | 49   | **2**       | 0.14110   | 6.08725   | 1.0      | 0.018301      |
| 9   | 50   | **1**       | 0.14030   | 6.12737   | 0.9      | 0.018146      |
| 10  | 51   | -           | 0.13126   | 6.61832   | 68.8     | 0.018061      |
| 11  | 52   | -           | 0.13393   | 6.46644   | 71.1     | 0.017863      |
| 12  | 53   | -           | 0.13293   | 6.52294   | 68.7     | 0.017672      |
| 13  | 54   | **38**      | 0.14001   | 6.14252   | 7.9      | 0.017718      |
| 14  | 55   | **56**      | 0.14049   | 6.11800   | 14.5     | 0.018073      |
| 15  | 56   | **144**     | 0.14001   | 6.14250   | 41.4     | 0.017759      |
| 16  | 57   | -           | 0.13914   | 6.18706   | 69.9     | 0.018240      |
| 17  | 58   | -           | 0.13724   | 6.28639   | 70.2     | 0.017865      |
| 18  | 59   | -           | 0.13531   | 6.39053   | 68.3     | 0.017736      |
| 19  | 60   | **4**       | 0.14068   | 6.10850   | 1.1      | 0.018429      |
| 20  | 61   | **0**       | 0.14127   | 6.07875   | 0.6      | 0.018301      |

### Multi-Run Results Summary

| Metric                              | Mean     | Std Dev  | Best       |
| ----------------------------------- | -------- | -------- | ---------- |
| Raw fitness (Σ\|error\|, 500 cases) | 6.26093  | 0.17909  | 5.96975    |
| Train MAE per fitness case (kW)     | 0.012522 | 0.000358 | 0.011939   |
| Train MAE as % of target range      | 8.58%    | 0.25%    | 8.18%      |
| Adjusted fitness `1/(1+raw)`        | 0.13780  | 0.00337  | 0.14348    |
| Successes (adj ≥ 0.14)              | -        | -        | **9 / 20** |
| Test MAE (kW, 40,321 rows)          | 0.018027 | 0.000251 | 0.017672   |
| Test MAE as % of target range       | 12.35%   | 0.17%    | 12.10%     |

### Best Individual

The best individual by training fitness was found in run 1 (seed 42), succeeding at generation 0, with a raw fitness of 5.96975 and an adjusted fitness of 0.14348, corresponding to a mean absolute error of **0.011939 kW per case** (8.18% of the target range):

```lisp
(lag_mean (lag_mean x_4 x_7) (lag_mean x_6 x_1))
```

This expression is simple and fully interpretable. It computes a nested average of four lag variables:

$$\hat{y}(t) = \frac{1}{2}\left(\frac{x_4 + x_7}{2}\right) + \frac{1}{2}\left(\frac{x_6 + x_1}{2}\right) = \frac{x_1 + x_4 + x_6 + x_7}{4}$$

In other words, it predicts tomorrow's load as the arithmetic mean of the loads four days ago, the same day last week (x₇ = 7-day lag), and two intermediate lags (x₁ = yesterday, x₆ = 6 days ago). Key observations:

- Weekly cycle: x₇ (the 7-day lag, same weekday) receives equal weight with the other lags, confirming the importance of the day-of-week pattern in residential demand.
- Domain-typed building blocks: The expression is built entirely from `lag_mean` nodes - the domain-typed function added in A2. This would not have been possible in A1's untyped function set without rediscovering the averaging structure from raw `add` and `mul` operations. The STGP type system guided the search toward physically meaningful compositions of load values.
- No ERCs needed: The best individual contains no ephemeral random constants. The GP discovered that a simple unweighted average of four lag days is sufficient, without needing scaling factors.
- Interpretability: The expression is algebraically equivalent to a 4-point moving average over lag days {1, 4, 6, 7}, which is a well-known estimator used in time-series forecasting. The GP rediscovered a principled forecasting method without being told to look for it.

### Comparison: A1 (Canonical GP) vs A2 (Structure-Based GP)

| Metric                     | A1 (Canonical GP) | A2 (STGP)    | Δ         |
| -------------------------- | ----------------- | ------------ | --------- |
| Mean Train MAE (kW)        | 0.012770          | 0.012522     | -0.000248 |
| Best Train MAE (kW)        | 0.012139          | 0.011939     | -0.000200 |
| Mean Test MAE (kW)         | 0.018123          | 0.018027     | -0.000096 |
| Best Test MAE (kW)         | 0.017664          | 0.017672     | +0.000008 |
| Mean Test MAE (% of range) | 12.41%            | 12.35%       | -0.07%    |
| Success rate (adj ≥ 0.14)  | 20.0% (4/20)      | 45.0% (9/20) | +25.0%    |
| Mean time per run (s)      | 14.6              | 41.6         | +27.0     |

#### Success Rate

The most important difference is the success rate: A2 achieves 9/20 (45%) versus A1's 4/20 (20%), more than doubling the probability of finding a solution within the 250-generation budget. Two runs (seeds 42 and 61) succeeded at generation 0, meaning the initial population already contained a successful individual, which reflects the benefit of a type-aware initialization that generates domain-meaningful trees from the very first generation.

#### Accuracy

Mean test MAE improved slightly from 0.018123 kW to 0.018027 kW (-0.5%). Best test MAE is essentially identical (0.017664 vs 0.017672 kW), differing by less than 1 watt. The accuracy improvement is modest, but the key gain is reliability: A2 finds good solutions more consistently across runs, as evidenced by the lower standard deviation (0.000251 vs 0.000350 kW).

#### Interpretability

A2's best individual `(lag_mean (lag_mean x_4 x_7) (lag_mean x_6 x_1))` is a simple, interpretable 4-point average. A1's best individual was a deeply nested expression involving `abs`, `sin`, `sqrt`, and `mul` with three lag variables - algebraically analyzable but not immediately interpretable in domain terms. A2's type system encouraged the search to build expressions from domain-meaningful operations, producing a result that a domain expert would recognize as sensible.

#### Runtime Cost

A2 is slower: 41.6 s/run versus 14.6 s/run for A1. This overhead comes from three sources: (1) type-aware population initialisation checks node type at every node created and rejects oversized trees; (2) type-safe crossover must scan parent 2 for all compatible donor nodes rather than picking a single random node; (3) the five new LOAD-typed functions are all arity-2, producing larger trees on average. The tradeoff is worthwhile given the substantial improvement in success rate.

### Discussion

#### Fitness Landscape

Raw fitness values converge across all twenty runs to a band of 5.97-6.62, with a cross-run standard deviation of 0.179 (coefficient of variation 2.9%). This is slightly tighter than A1's range (6.07-6.79, CoV 3.4%), suggesting the STGP type constraints guide the search to a more consistent region of the fitness landscape.

#### Domain-Typed Functions

The domain-typed functions (lag_mean, lag_max, lag_min, lag_delta, lag_ratio) appear prominently in the evolved expressions. The best individual uses only `lag_mean`, and most successful runs produce expressions dominated by `lag_mean` and `lag_min` trees. This makes physical sense: load forecasting is well-served by smooth interpolation and lower-bound estimation of historical load readings. The GP discovered this preference without being explicitly told.

#### Effect of the STGP Type System

The type system constrains the search in two ways that improve performance: (1) it prevents semantically meaningless combinations such as applying `lag_mean` to an ERC or the output of `sin`, which waste crossover and mutation events; (2) it channels the search toward combinations of load values, which are the most informative inputs for this problem. The doubled success rate is the empirical evidence that these constraints are helpful rather than harmful.

#### Lag Variable Selection

The best individual uses x₁, x₄, x₆, x₇. The 7-day lag (x₇) appears in most high-fitness individuals across both A1 and A2, confirming the dominance of the weekly periodicity in residential demand. The inclusion of x₄ (4 days ago, mid-week) is specific to A2 and may reflect the GP's discovery that the mid-week load provides additional signal about weekday vs. weekend transitions.

#### Generalisation

The mean train MAE of 0.012522 kW versus mean test MAE of 0.018027 kW gives a train-to-test ratio of approximately 1.44×, similar to A1's ratio of 1.42×. Both implementations show a comparable degree of overfitting to the 500-case training sample. Part of the gap is also attributable to the COVID-19 lockdown period (March-September 2020) present in the test set but not in the training data, which altered residential consumption patterns in ways the GP could not have observed during training.

---

## 11. Runtimes

### Per-Run and Total Times

| Metric                  | Value                                          |
| ----------------------- | ---------------------------------------------- |
| Mean time per run       | 41.6 s                                     |
| Total for 20 runs       | 832.0 s (~13.9 min)                        |
| Max evaluations per run | 250 gens × 500 individuals = 125,000 max evals |
| Fastest run             | 0.6 s (run 20, seed 61 - succeeded at gen 0)   |
| Slowest run             | 77.3 s (run 6, seed 47 - ran all 250 gens)     |

### Breakdown of Runtime Factors

| Factor                        | Effect                                                                |
| ----------------------------- | --------------------------------------------------------------------- |
| `POP_SIZE = 500`              | 500 tree evaluations per generation                                   |
| `N_FITNESS_CASES = 500`       | Each evaluation scores the tree on 500 input vectors                  |
| `MAX_GENERATIONS = 250`       | Up to 250 × 500 = 125,000 evaluations per run                         |
| Vectorised NumPy evaluation   | All 500 fitness cases computed in a single array pass                 |
| Early stopping                | 9 runs terminated early, saving significant time                      |
| Type-aware operator overhead  | Crossover scans parent 2 for compatible nodes before selecting a donor|
| Larger average tree size      | LOAD-typed arity-2 functions produce denser trees; copies are slower  |

The increase from A1's 14.6 s/run to A2's 41.6 s/run is primarily due to the type-aware operators and the larger trees that the LOAD-typed arity-2 functions produce. Despite the slowdown, the 9/20 success rate means that only 9 runs ran close to the full budget; the remaining 11 terminated early, keeping the total experiment time under 14 minutes.

### Computational Effort

The `computational_effort()` function in `reporting.py` computes the Koza (1992) computational effort metric. For each generation $i$:

$$I(z, M, i) = R(z, M, i) \times M \times (i + 1)$$

where:
- $z = 0.99$ (target probability of at least one success across all runs)
- $M = 500$ (population size)
- $p(i)$ = empirical fraction of runs that found a solution by generation $i$
- $R(z, M, i) = \lceil \log(1-z) / \log(1-p(i)) \rceil$ = minimum runs needed for 99% success probability

In this experiment, with 9 of 20 runs succeeding (two at generation 0):

$$\min_i \; I(0.99,\, 500,\, i) = \mathbf{22{,}000} \quad \text{at generation } i^* = 0$$

At generation 0, two runs had already succeeded (seeds 42 and 61), giving $p(0) = 2/20 = 0.10$. This yields $R = \lceil \log(0.01) / \log(0.90) \rceil = \lceil 43.7 \rceil = 44$ independent runs needed for 99% confidence, and $I = 44 \times 500 \times 1 = 22{,}000$ total evaluations. This is dramatically lower than A1's $I^* = 319{,}000$, a reduction of more than 14×. The type-aware initialisation producing high-quality individuals in the very first generation is the key driver of this improvement.

---

## References

Koza, J. R. (1992). _Genetic Programming: On the Programming of Computers by Means of Natural Selection_. MIT Press, Cambridge, MA.
