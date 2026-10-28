#  Genetic Programming for Electricity Load Forecasting
---

## 1. Description of the Dataset

### Source and Overview

The dataset used in this project is the UK Residential Energy Dataset (2014–2020), a time-series dataset of residential electricity consumption recorded at 15-minute intervals. The data is stored in a CSV file and contains 201,604 readings spanning from 31 December 2014 to 30 September 2020 which is a period of approximately 5.75 years (2,100 days). The dataset provides a comprehensive view of residential energy behavior across multiple seasons and years, making it good for learning and evaluating time-series forecasting models.

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

Only the `Electricity_load` column is used because this is treated as a time-series problem: can load at time _t_ be predicted from lagged load values at earlier times, without relying on external covariates? While including covariates such as temperature, humidity, or price would likely improve accuracy, doing so would combine the GP's symbolic regression capability with feature engineering. Using only lagged load values keeps the search space clean and ensures that results are attributable to the GP alone.

### Data Preprocessing

No pre processing was required as the dataset contains no missing values across all 201,604 rows. The `utc_timestamp` column was converted to a Python `datetime` object using the format `'%d/%m/%Y %H:%M'` to enable chronological ordering. No normalization or scaling was applied to the target variable, as the GP operates on raw load values and the fitness measure (sum of absolute errors) is expressed in the same units (kW), making the threshold directly readable.

### Train/Test Split

The dataset is divided chronologically into a training set (first 80%) and a held-out test set (remaining 20%):

| Partition | Rows    | Date Range                          |
| --------- | ------- | ----------------------------------- |
| Training  | 161,283 | 31/12/2014 - 07/08/2019 23:30       |
| Test      | 40,321  | 07/08/2019 23:45 - 30/09/2020 23:45 |

The split is chronological rather than random. Randomly splitting temporal data would leak future information into training, whereas a chronological split correctly simulates real deployment conditions: train on the past, evaluate on the unseen future. The test set covers over 14 months of unseen data, providing a good evaluation of generalization.

### Input Variable Construction

Seven lagged input features are constructed, each representing the electricity load at the same time-of-day on each of the previous seven days. Since there are 96 readings per day, the lag for day _k_ is _k_ × 96 time steps back:

> x₁ = load at t − 96, x₂ = load at t − 192, x₃ = load at t − 288, x₄ = load at t − 384, x₅ = load at t − 480, x₆ = load at t − 576, x₇ = load at t − 672

This design reflects the periods in residential electricity consumption at both the 24-hour scale (sleep cycles, cooking, heating) and the 7-day scale (weekday vs. weekend patterns). By anchoring each lag to the same time-of-day, the input features create directly comparable historical snapshots, giving the GP access to exactly one full weekly cycle of history. The lag construction also removes the need for any seasonal differencing, as the same-time-of-day anchoring removes the daily periodic component from the regression problem.

### Fitness Cases

Because evaluating all 161,283 training rows each generation would be computationally expensive, a random sample of 500 rows is drawn from the training set once at the start of each GP run and fixed for all generations within that run. This keeps the fitness signal consistent: if the sample changed each generation, an individual's fitness could change for reasons unrelated to its true quality, introducing noise into the selection process and destabilizing the fitness landscape. The 500-case sample is large enough to provide a representative signal (covering a wide variety of time-of-day and seasonal patterns) while keeping per-generation evaluation time under one second.

---

## 2. Performance Metric

### Fitness Measure: The Koza (1992) Four-Step Chain

This implementation adopts the complete four-level fitness transformation as described by Koza (1992), progressing from a raw error measure to a normalised probability suitable for selection. The chain is implemented in `fitness.py`.

| Step         | Formula                   | Direction                     |
| ------------ | ------------------------- | ----------------------------- |
| Raw          | `Σ                        | ŷ − y                         |
| Standardised | = raw (already ≥ 0)       | lower = better                |
| Adjusted     | `1 / (1 + raw)`           | higher = better, ∈ (0, 1]     |
| Normalised   | `adjusted_i / Σ adjusted` | sums to 1.0 across population |

#### Step 1 - Raw Fitness

Raw fitness is defined as the sum of absolute errors across all 500 fitness cases:

$$r(i) = \sum_{j=1}^{N} \left| \hat{y}_j - y_j \right|$$

where $\hat{y}_j$ is the GP tree's prediction for fitness case $j$ and $y_j$ is the actual electricity load. A raw fitness of 0 represents a perfect predictor. Lower values are better.

**Why sum of absolute errors (not squared)?** SAE is less sensitive to outlier fitness cases than sum of squared errors and produces values in the same units as the target variable (kW), making thresholds directly interpretable. For example, the success threshold of `SUCCESS_THRESHOLD = 0.14` (adjusted fitness) corresponds to a raw fitness of $1/0.14 - 1 \approx 6.143$, which equals a mean absolute error of $6.143 / 500 \approx 0.01229$ kW per case (approximately 8.42% of the full target range of 0.146 kW). This is a concrete, physically meaningful criterion.

#### Step 2 - Standardized Fitness

Standardized fitness is defined such that lower values are better and 0 is optimal. Since sum of absolute errors already satisfies this property, standardized fitness is set equal to raw fitness:

$$s(i) = r(i)$$

#### Step 3 - Adjusted Fitness

Adjusted fitness maps standardized fitness to the range **(0, 1]**, where **higher is better** and 1 represents a perfect individual:

$$a(i) = \frac{1}{1 + s(i)}$$

This transformation compresses the unbounded raw fitness onto a bounded scale. It is particularly important for fitness-proportionate selection because it ensures that even very poor individuals receive a small but non-zero selection probability, preventing complete starvation of weaker individuals early in the run. The adjusted fitness is used for the success check because it provides an absolute, population-independent criterion: an individual's adjusted fitness of 0.20 means the same thing regardless of how good or bad the rest of the population is.

#### Step 4 - Normalized Fitness

Normalized fitness expresses each individual's adjusted fitness as a fraction of the total adjusted fitness across the entire population:

$$n(i) = \frac{a(i)}{\sum_{j=1}^{M} a(j)}$$

All normalized fitness values across a population of size $M$ sum to exactly 1.0, making them directly interpretable as selection probabilities for roulette-wheel (fitness-proportionate) selection. 

### Edge Case Handling

Numerical edge cases that would violate the closure property are handled in `primitives.py`:

- **Division by zero:** Protected division (`pdiv`) returns 1.0 when the denominator's absolute value falls below 1×10⁻¹⁰, and clips the result to ±10⁶ otherwise.
- **Square root of negative numbers:** Protected square root (`psqrt`) computes `sqrt(|a|)`, ensuring a real-valued result for any input.
- **Overflow and NaN propagation:** A final safety pass (`_safe`) replaces any remaining NaN or ±Inf values with ±10⁶ and clips all outputs to the range $[−10⁶, +10⁶]$.
- **Zero total adjusted fitness:** If all individuals in a population produce degenerate outputs and their total adjusted fitness falls below 1×10⁻¹⁵, the normalization denominator is set to 1.0, giving each individual an equal selection probability and preventing division by zero in the normalization step.

---

## 3. Representation: Terminal and Function Sets

Each GP individual is represented as a parse tree - a ordered tree in which internal nodes perform operations and leaf nodes supply values. This is the canonical representation for symbolic regression. Its advantage over implicit approaches such as neural networks is interpretability: a parse tree produces an explicit, human-readable formula that can be inspected, simplified, and reasoned about. For example, the best individual found is `(sqrt (mul (sqrt (mul (abs x_6) (abs (abs x_1)))) (sin (sqrt (mul (abs (abs (abs (abs x_4)))) (abs (neg x_7)))))))`, which is a algebraically analyzable expression.

### Terminal Set

The terminal set consists of two categories of leaf node:

|Terminal|Description|
|---|---|
|x₁, x₂, …, x₇|Input variables: lagged electricity load values (one per previous day at the same time)|
|ERC|Ephemeral Random Constant: a floating-point value drawn uniformly from [−5.0, 5.0] at tree creation time and permanently embedded in the node|

**Why Ephemeral Random Constants (ERCs)?** The GP needs to discover not just the _structure_ of a formula but also appropriate numerical coefficients and offsets. ERCs allow the evolutionary search to find these constants without requiring them to be pre-specified. The range $[−5, +5]$ was chosen to be wide enough to allow meaningful scaling of the load values (which lie in $[0.031, 0.177]$) without being so wide as to produce constants that overwhelm the small magnitude of the target.

### Function Set

| Function | Symbol | Arity | Description                                                   |
| -------- | ------ | ----- | ------------------------------------------------------------- |
| `add`    | +      | 2     | Addition                                                      |
| `sub`    | −      | 2     | Subtraction                                                   |
| `mul`    | ×      | 2     | Multiplication                                                |
| `div`    | ÷      | 2     | **Protected** division - returns 1.0 when denominator ≈ 0     |
| `sqrt`   | √      | 1     | **Protected** square root - computes √\|a\|                   |
| `sin`    | sin    | 1     | Sine (argument clipped to $[−10⁶, +10⁶]$ before evaluation)   |
| `cos`    | cos    | 1     | Cosine (argument clipped to $[−10⁶, +10⁶]$ before evaluation) |
| `neg`    | −      | 1     | Unary negation                                                |
| `abs`    | \|·\|  | 1     | Absolute value                                                |

**Why this function set?** The four arithmetic operations (`add`, `sub`, `mul`, `div`) form a complete algebraic basis capable of expressing any rational function. `sqrt`, `sin`, and `cos` introduce non-linearity and the potential for periodic components - important for a target variable with strong daily and weekly cycles. `neg` and `abs` allow sign manipulation and guarantee non-negative outputs. Protected versions of `div` and `sqrt` guarantee closure - every tree always returns a finite real number regardless of inputs, which is essential for safe population-wide vectorized evaluation.

### Closure Property

Every function in the function set accepts any real number as input and returns a real number as output. Protected operations guarantee this: `div` handles the denominator-zero case explicitly; `sqrt` takes the absolute value of its argument before computing the root; clipping to ±10⁶ and the NaN/Inf safety pass ensure no non-finite values propagate upward through the tree.

### Sufficiency Property

The combination of the four arithmetic operations, trigonometric functions, square root, absolute value, and seven lagged input variables is sufficient to approximate a wide class of time-series patterns, including periodic, trend-following, and autoregressive behavior. The inclusion of ERCs allows the GP to fit the amplitude and offset of discovered expressions. Sufficiency does not guarantee that the GP _will_ find a perfect solution, but it ensures the solution space contains good approximations.

### Tree Representation and Printability

Each individual is an instance of the `Individual` class (defined in `tree.py`), containing a root `Node` object. Each `Node` stores its `value` (function name or terminal value) and a list of `children`. The tree can be rendered in two formats:

- **Lisp-style prefix notation** (for human inspection): e.g., `(add x_1 (mul x_2 3.0))`
- **Canonical infix expression** (for duplicate detection): e.g., `(x_1+(x_2*3.0))`

---

## 4. Initial Population Generation

### Method: Ramped Half-and-Half

The initial population is generated using Ramped Half-and-Half, the method recommended by Koza (1992) as producing the best initial diversity. A population generated entirely with the Full method would produce trees of identical shape at each depth, leading to structural redundancy; Grow alone tends to produce very small trees. Ramping across depths 2–8 and alternating between the two ensures the initial population contains trees of varied size, shape, and complexity - from small expressions at depth 2 to densely structured Full trees at depth 8. The implementation is in `population.py` and operates as follows:

1. Depth range: Tree depths cycle through `MIN_INIT_DEPTH = 2` to `MAX_INIT_DEPTH = 8` inclusive (seven depth levels). Individuals are assigned depths by `depth = depths[idx % 7]` as they are added. With 500 individuals across 7 levels (500 = 71 × 7 + 3), the first three depths (2, 3, 4) receive 72 slots and the remaining four (5, 6, 7, 8) receive 71 slots each — a near-uniform distribution.

2. Half-and-Half within each depth: Across each group of 7 slots (one per depth level), the generation method alternates between Full and Grow in whole batches, yielding approximately half Full and half Grow trees at each depth:
    
    - _Full method:_ Every path from root to leaf has exactly the target depth. All internal positions are function nodes; only the nodes at the target depth are terminals, producing dense, maximally bushy trees.
    - _Grow method:_ At each internal position, a function or terminal is chosen uniformly at random from the combined set. Terminals can appear at any depth, producing smaller and more structurally varied trees.

3. Duplicate rejection: Each generated tree is converted to its canonical infix expression string and compared against all previously accepted individuals. Rejecting duplicates guarantees that every slot in the initial population is occupied by a structurally distinct individual, ensuring the population begins at 100% variety - the maximum possible genetic diversity for a population of this size. This is important because low initial diversity is one of the primary causes of premature convergence identified in the GP literature.


---

## 5. Fitness Function and Fitness Evaluation

### The `evaluate_fitness` Function

The fitness of a single individual is computed by `evaluate_fitness(ind, X, Y)` in `fitness.py`. This function:

1. Vectorised tree evaluation: Passes the individual's parse tree and the full dictionary of input arrays `X` to `eval_tree()` in `primitives.py`, which evaluates the tree recursively using NumPy array operations. Each node operation (e.g., `add`) calls `np.add(left_array, right_array)`, computing all 500 fitness cases in a single loop. This is faster than evaluating one case at a time in Python.

2. Raw fitness: Computes fitness as the sum of absolute differences between the tree's predictions and the actual load values in `Y`.

3. Fitness storage: Stores the raw, standardized, and adjusted fitness values on the individual object. Normalized fitness is computed separately once all individuals in the population have been evaluated.


### Population-Level Evaluation

`evaluate_population(pop, X, Y)` is called once per generation. It iterates over every individual in the population, calling `evaluate_fitness` for each, and then computes normalised fitness values as each individual's adjusted fitness divided by the population total. This ensures that every individual's fitness is up to date before selection begins, and that normalised fitness values across the population sum to exactly 1.0.

### Fitness Cases and Consistency

The same 500 fitness cases are used for every generation within a run. If cases were re-sampled each generation, an individual might appear to improve or regress solely because the sample changed, not because the individual genuinely improved. Fixed cases make fitness a stable, reproducible property of the individual. The 500 cases are sampled once at the start of each run using the run's fixed random seed, ensuring both within-run consistency and cross-run reproducibility.

---

## 6. Selection Method

### Method: Fitness-Proportionate Selection (Roulette Wheel)

Selection is implemented by the `fps(pop)` function in `fitness.py` using the standard roulette-wheel method. Fitness-Proportionate Selection (FPS) is the natural and required pairing for the normalized fitness measure: normalized fitness values sum to 1.0 and directly form a valid probability distribution, so FPS samples from that distribution with no additional scaling or transformation.

Under FPS, an individual with twice the normalized fitness is twice as likely to be selected. This gives proportional selection pressure that favors fitter individuals while still preserving a non-zero survival chance for weaker ones - important for maintaining diversity and preventing premature convergence. Tournament selection, operates on rank within a small group and does not require a probability distribution, meaning it would ignore the normalized fitness values entirely, which would be a waste given the fitness transformation pipeline.

The algorithm proceeds as follows:

1. Draw a random number _r_ uniformly from $[0, 1)$.
2. Walk through the population in order, accumulating each individual's normalized fitness into a running total.
3. Return the first individual whose cumulative total reaches or exceeds _r_.
4. If floating-point rounding causes no individual to be selected (a rare edge case), return the last individual as a safety fallback.

---

## 7. Genetic Operators

All three core GP operators are implemented in `operators.py`. Their application rates are configurable constants in `config.py` that sum to 1.0.

|Operator|Rate|Function|
|---|---|---|
|Crossover|0.80|`crossover(p1, p2)`|
|Mutation|0.10|`mutate(ind)`|
|Reproduction|0.10|`reproduce(ind)`|

### 7.1 Reproduction - `reproduce(ind)`

The individual is deep-copied and placed unchanged into the next generation. No modification is made to the tree structure, fitness values, or constants. Reproduction preserves high-quality individuals across generations and contributes to selection pressure without introducing the risk of destroying good solutions. It also reduces wasted computation: a reproduced individual's fitness does not need to be re-evaluated because its tree is unchanged.

### 7.2 Subtree Crossover - `crossover(p1, p2)`

Subtree crossover is the primary search mechanism, assembling new individuals from proven building blocks by recombining subtrees from two fit parents. The rate of 80% follows the conventional Koza default. The operator creates one offspring as follows:

1. Deep-copy parent 1 to form the offspring.
2. Select a random cut point in the offspring from positions 1 to n−1 (non-root), ensuring the offspring is a genuine mix of both parents.
3. Select a random donor subtree root anywhere in parent 2.
4. Replace the subtree rooted at the cut point in the offspring with a copy of the donor subtree.
5. If the resulting tree exceeds `MAX_TREE_DEPTH = 17` or `MAX_TREE_NODES = 100`, discard the offspring and return a copy of parent 1 instead, preventing unbounded tree growth (bloat).

Nodes are indexed in depth-first, left-to-right order. The `_get` helper navigates to any node by index, returning the node itself, its parent, and the child position so that the replacement can be performed in-place.

### 7.3 Subtree Mutation - `mutate(ind)`

Subtree mutation reintroduces random genetic material that cannot arise from recombining existing population members. Without it, the population's collective gene pool would gradually homogenize as crossover repeatedly reuses the same subtrees. At a rate of 10%, mutation acts as a diversity maintenance mechanism without dominating the search. The operator proceeds as follows:

1. Deep-copy the individual.
2. Select a random mutation point anywhere in the tree (including the root).
3. Generate a fresh random subtree using the Grow method with depth chosen uniformly from \[1, `MAX\_INIT\_DEPTH`\].
4. Replace the subtree at the mutation point with the new subtree.
5. If the result exceeds the depth or node count limits, return the original individual unchanged.

### Size and Depth Enforcement

Both crossover and mutation enforce hard limits of `MAX_TREE_DEPTH = 17` and `MAX_TREE_NODES = 100`. These caps address the GP "bloat" phenomenon: because crossover has a statistical bias toward selecting larger subtrees as donors, average tree size tends to grow without intervention. Hard caps are simpler and computationally cheaper than parsimony pressure (penalizing tree size in the fitness function) and effectively prevent unbounded growth by discarding oversized offspring outright.

---

## 8. Experimental Setup

### Control Model

A **generational control model** with single-individual elitism is used, following Koza (1992). The generational model replaces the entire population each generation, providing a clean separation between evaluation and reproduction. Elitism ensures that the best individual from the current generation is always copied into the next generation. Without elitism, crossover or mutation could accidentally destroy the best-ever solution before it contributes offspring, whereas with elitism the trajectory of best fitness is guaranteed to be monotonically non-increasing in raw error.

In each generation:

- Every individual in the current population is replaced by offspring produced through crossover, mutation, or reproduction.
- The single best individual from the current generation is guaranteed a copy in the next generation.

### Termination Criterion

A run terminates when **either** of the following conditions is met:

1. The maximum number of generations (`MAX_GENERATIONS = 250`) is reached.
2. The best individual's adjusted fitness reaches or exceeds the success threshold (`SUCCESS_THRESHOLD = 0.14`), corresponding to a mean absolute error per fitness case of at most 0.01229 kW (8.42% of the full target range).

### Complete Parameter Configuration

All parameters are centralized in `config.py`. No values are hardcoded elsewhere in the codebase.

|Parameter|Value|Description|
|---|---|---|
|`DATA_PATH`|`data/Residential_Energy_Dataset_UK-2014-2020.csv`|CSV path|
|`N_DAYS_BACK`|7|Number of daily lag inputs (x₁ … x₇)|
|`TRAIN_RATIO`|0.80|First 80% of data used for training|
|`N_FITNESS_CASES`|500|Training samples per run|
|`POP_SIZE`|500|Individuals per generation|
|`MIN_INIT_DEPTH`|2|Minimum tree depth at initialisation|
|`MAX_INIT_DEPTH`|8|Maximum tree depth at initialisation|
|`MAX_TREE_DEPTH`|17|Hard depth cap on offspring|
|`MAX_TREE_NODES`|100|Hard node-count cap on offspring|
|`ERC_MIN`|−5.0|Lower bound for ephemeral constants|
|`ERC_MAX`|+5.0|Upper bound for ephemeral constants|
|`CROSSOVER_RATE`|0.80|Subtree crossover probability|
|`MUTATION_RATE`|0.10|Subtree mutation probability|
|`REPRODUCTION_RATE`|0.10|Reproduction probability|
|`MAX_GENERATIONS`|250|Maximum generations per run|
|`SUCCESS_THRESHOLD`|0.14|Early-stop adjusted fitness threshold|
|`N_RUNS`|20|Number of independent runs|
|`BASE_SEED`|42|Run _k_ uses seed BASE_SEED + _k_|

### Reproducibility

Each run uses a fixed random seed of `BASE_SEED + run_index` applied to both Python's built-in `random` module and NumPy's random generator. This guarantees that every run can be reproduced identically by re-running the code with the same configuration. Seeds are reported in output alongside each run's results. All hyperparameters are versioned in `config.py`

### Technical Specifications

| Component          | Value                                                |
| ------------------ | ---------------------------------------------------- |
| Python             | 3.12.3                                               |
| NumPy              | 1.26.4                                               |
| OS                 | Linux WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2) |
| External libraries | NumPy only                                           |
| GP implementation  | From scratch                                         |

Twenty independent runs are performed with different random seeds (seeds 42–61). A single GP run is stochastic - its outcome depends heavily on the initial population and the sequence of operator applications - so twenty runs provide enough samples to compute meaningful statistics (mean, standard deviation) and to apply the Koza (1992) computational effort metric, which requires an empirical estimate of success probability per generation.

---

## 9. Results and Discussion

### Multi-Run Results Summary

Twenty independent runs were performed (seeds 42–61, up to 250 generations each, `POP_SIZE = 500`, `N_FITNESS_CASES = 500`):

| Metric                              | Mean     | Std Dev  | Best       |
| ----------------------------------- | -------- | -------- | ---------- |
| Raw fitness (Σ\|error\|, 500 cases) | 6.38515  | 0.21838  | 6.06931    |
| Train MAE per fitness case (kW)     | 0.012770 | 0.000437 | 0.012139   |
| Train MAE as % of target range      | 8.74%    | 0.30%    | 8.32%      |
| Adjusted fitness `1/(1+raw)`        | 0.13552  | 0.00399  | 0.14146    |
| Successes (adj ≥ 0.14)              | -        | -        | **4 / 20** |
| Cross-run coefficient of variation  | 3.42%    | -        | -          |
| Test MAE (kW, 40,321 rows)          | 0.018123 | 0.000350 | 0.017664   |
| Test MAE as % of target range       | 12.41%   | 0.24%    | 12.10%     |

### Per-Run Summary

| Run | Seed | Success Gen | Final Adj | Final Raw | Time (s) | Test MAE (kW) |
|-----|------|-------------|-----------|-----------|----------|---------------|
| 1   | 42   | **14**      | 0.14001   | 6.14241   | 0.6      | 0.018513      |
| 2   | 43   | -           | 0.13599   | 6.35328   | 13.9     | 0.018398      |
| 3   | 44   | -           | 0.13054   | 6.66033   | 16.0     | 0.018192      |
| 4   | 45   | -           | 0.13853   | 6.21866   | 15.2     | 0.018186      |
| 5   | 46   | -           | 0.13288   | 6.52564   | 16.2     | 0.017919      |
| 6   | 47   | -           | 0.13310   | 6.51299   | 14.3     | 0.017738      |
| 7   | 48   | -           | 0.13040   | 6.66870   | 37.2     | 0.017664      |
| 8   | 49   | **21**      | 0.14087   | 6.09893   | 1.0      | 0.018548      |
| 9   | 50   | -           | 0.13524   | 6.39407   | 13.0     | 0.018958      |
| 10  | 51   | -           | 0.12837   | 6.79017   | 15.6     | 0.018008      |
| 11  | 52   | -           | 0.13182   | 6.58630   | 15.1     | 0.017983      |
| 12  | 53   | -           | 0.13136   | 6.61263   | 18.6     | 0.017794      |
| 13  | 54   | -           | 0.13903   | 6.19275   | 16.8     | 0.017749      |
| 14  | 55   | -           | 0.13948   | 6.16947   | 18.0     | 0.017789      |
| 15  | 56   | -           | 0.13793   | 6.25010   | 16.3     | 0.018195      |
| 16  | 57   | -           | 0.13500   | 6.40733   | 13.9     | 0.018311      |
| 17  | 58   | -           | 0.13578   | 6.36494   | 31.9     | 0.017800      |
| 18  | 59   | -           | 0.13254   | 6.54490   | 16.7     | 0.017859      |
| 19  | 60   | **11**      | 0.14005   | 6.14012   | 0.5      | 0.018514      |
| 20  | 61   | **36**      | 0.14146   | 6.06931   | 2.0      | 0.018341      |

### Best Individual

The best individual was found in run 20 (seed 61), succeeding at generation 36, with a raw fitness of 6.06931 and an adjusted fitness of 0.14146, corresponding to a mean absolute error of **0.01214 kW per case** (8.32% of the target range):

```lisp
(abs (abs (sqrt (mul (sin (sqrt (mul (abs (abs (sin x_6))) (abs (abs (sin (abs x_1))))))) (abs (sin (sin (abs (abs x_7)))))))))
```

Key observations about this expression:

- Three of seven lag variables are used: x₁ (1-day), x₆ (6-day), and x₇ (7-day). The absence of x₂–x₅ suggests the GP found that the most recent day and the tail of the preceding week carry the dominant predictive signal.

- x₇ (the 7-day lag) appears prominently, confirming the importance of the weekly cycle - weekday vs. weekend consumption patterns are a well-established driver of residential demand.

- The outer `sqrt(mul(...))` pattern approximates a geometric mean of lagged loads, a natural robust estimator for a positive, bounded variable. The GP rediscovered this structural property without being told.

- The two branches of the `mul` combine a `sin`-transformed product of x₁ and x₆ with a `sin`-transformed x₇. For small arguments in $[0, 0.177]$, `sin(x) ≈ x`, so the trigonometric wrappers approximately reduce to square-root scaling of the lag products, making the expression effectively a weighted geometric average of three lagged loads.

- Nested `abs` wrappers: The repeated `abs` calls are functionally redundant (applying `abs` twice is identical to applying it once), reflecting GP's tendency to accumulate neutral structure. The expression could be simplified to `sqrt(mul(sin(sqrt(mul(|sin x_6|)(|sin x_1|)))), |sin(sin x_7)|)` with equivalent behavior.

### Baseline Comparison

Two baselines are evaluated on the full held-out test set (40,321 rows) to contextualize GP performance:

| Predictor              | Test MAE (kW) | MAE as % of target range |
| ---------------------- | ------------- | ------------------------ |
| Mean predictor         | 0.021826      | 14.95%                   |
| Persistence (x₁ only) | 0.022305      | 15.28%                   |
| GP mean (all 20 runs)  | 0.018123      | 12.41%                   |
| GP best (run 7)        | 0.017664      | 12.10%                   |

#### Mean Predictor

Predicts every test point as the training mean (0.0946 kW), ignoring all lag structure. This is the weakest possible baseline with an MAE of 0.021826 kW (14.95% of the target range).

#### Persistence (x₁)

Predicts y(t) = x₁, i.e., the same time-of-day load from the previous day. This requires no model at all - just one memory lookup. Its MAE of 0.022305 kW is slightly *worse* than the mean predictor, which is somewhat surprising and suggests the single-day lag introduces noise on the test period (which covers August 2019 - September 2020 and includes the COVID-19 lockdown, a period when day-to-day load patterns were more erratic than the pre-lockdown training data).

#### GP Performance vs Baselines

The GP mean test MAE of 0.018123 kW is 17% better than the mean predictor and 19% better than persistence, confirming that the evolved expressions capture real predictive structure beyond what trivially simple predictors can achieve. The best GP individual (run 7, seed 48) achieves 0.017664 kW - a 19.1% improvement over the mean baseline.

### Discussion

#### Fitness Landscape

Raw fitness values (Σ|error| over 500 cases) converge across all twenty runs to a band of 6.07–6.79, with a cross-run coefficient of variation of 3.42%. This low variability indicates the GP finds similar-quality solutions across seeds, consistent with a smooth search landscape without strongly deceptive local optima.
#### Success Rate

Four of twenty runs (20%) reached the success threshold of adjusted fitness ≥ 0.14. The successful runs vary in when they converge: three find a solution almost immediately (gens 11, 14, 21), while one requires 36 generations. This fast convergence or none pattern suggests the population either discovers a high-quality structural building block in the early generations or settles into a local optimum from which escape is difficult within the 250-generation budget.
#### Lag variable effectiveness.

The seven lag variables correspond to the same time-of-day on each of the seven previous days. The GP's selection of x₇ across multiple successful runs confirms the importance of the weekly cycle in residential demand. The best individual uses only x₁, x₆, and x₇, suggesting the most recent day and the end of the preceding week provide the strongest predictive signal.
#### Operator balance.

The 0.80 crossover rate is expansive within the search, aligning with Koza's recommendations. This ensures the search is primarily driven by recombination of existing building blocks. Mutation at 0.10 maintains diversity and prevents premature convergence by reintroducing novel structures. The elitism scheme ensures the best-ever fitness trajectory is monotonically non-increasing.
#### Bloat control.

The hard limits of `MAX_TREE_DEPTH = 17` and `MAX_TREE_NODES = 100` constrain tree growth. The best individual is compact (22 nodes, depth 11), while some unsuccessful runs accumulated larger trees (approaching the 100-node cap) by generation 250 without achieving proportionally better fitness - consistent with the GP bloat phenomenon.

#### Generalisation.

Training fitness is evaluated on only 500 randomly sampled rows fixed at the start of each run, while test MAE is evaluated post-hoc on all 40,321 held-out rows. The mean train MAE of 0.012770 kW is substantially lower than the mean test MAE of 0.018123 kW - a ratio of approximately 1.42×. This gap indicates that GP solutions overfit to some degree to the 500-case training sample: the expressions that score best on a small fixed sample do not transfer perfectly to the full test distribution. Part of this gap is also attributable to distribution shift - the test set covers August 2019 - September 2020 and includes the COVID-19 lockdown, a period of atypical residential consumption not represented in the pre-lockdown training data. Despite the gap, test MAE remains well below both baselines (17–19% improvement), confirming that the evolved expressions capture genuine predictive structure.

---

## 10. Runtimes

### Per-Run and Total Times

| Metric                    | Value                                         |
| ------------------------- | --------------------------------------------- |
| Mean time per run         | **14.6 s**                                    |
| Total for 20 runs         | **292.2 s (~4.9 min)**                        |
| Max generations per run   | 250 gens × 500 individuals = 125,000 max evals|
| Fastest run               | 0.5 s (run 19, succeeded at gen 11)           |
| Slowest run               | 38.8 s (run 7, ran all 250 gens)              |

### Breakdown of Runtime Factors

| Factor                      | Effect                                                               |
| --------------------------- | -------------------------------------------------------------------- |
| `POP_SIZE = 500`            | 500 tree evaluations per generation                                  |
| `N_FITNESS_CASES = 500`     | Each evaluation scores the tree on 500 input vectors                 |
| `MAX_GENERATIONS = 250`     | Up to 250 × 500 = 125,000 evaluations per run                        |
| Vectorised NumPy evaluation | speedup over Python-loop evaluation                                  |
| Early stopping              | 4 runs terminated early (gens 11, 14, 21, 36), saving time           |
| Operator overhead           | Crossover and mutation include tree traversal and copy steps         |

The runtime is dominated by vectorized NumPy tree evaluation. Each individual is scored on all 500 fitness cases in a single recursive pass through the tree using NumPy array operations. For example, an `add` node does not loop over 500 cases; it calls `np.add(left_array, right_array)` once, computing all 500 results at the C level. This is faster than a Python loop over cases. Without vectorization, the estimated runtime would be longer, rendering the 20-run experiment impractical.

The wide spread in per-run times (0.5 s to 38.8 s) reflects the effect of early stopping: runs that found a successful individual in the first few generations completed in under 2 seconds, while runs that ran to the 250-generation limit took 14–39 seconds depending on tree size growth (larger trees are slower to copy and traverse during crossover and mutation).

### Computational Effort

The `computational_effort()` function in `reporting.py` computes the Koza (1992) computational effort metric post-hoc. For each generation $i$:

$$I(z, M, i) = R(z, M, i) \times M \times (i + 1)$$

where:

- $z = 0.99$ (target probability of finding at least one solution)
- $M = 500$ (population size)
- $p(i)$ = empirical fraction of runs that found a solution by generation $i$
- $R(z, M, i) = \lceil \log(1-z) / \log(1-p(i)) \rceil$ = minimum runs needed for 99% success probability

The generation $i^*$ that minimizes $I$ is the computational sweet spot - the point at which the cost of running multiple shorter independent runs is minimized.

In this experiment, with 4 of 20 runs succeeding:

$$\min_i \; I(0.99,\, 500,\, i) = \mathbf{319{,}000} \quad \text{at generation } i^* = 22$$

At generation 22, three runs had already succeeded (seeds 42, 49, 60 - finding solutions at generations 14, 21, and 11 respectively), giving $p(22) = 3/20 = 0.15$. This yields $R = \lceil \log(0.01)/\log(0.85) \rceil = 29$ independent runs needed for 99% confidence, and $I = 29 \times 500 \times 22 = 319{,}000$ total evaluations. Running more generations beyond $i^* = 22$ increases the per-run cost faster than the improvement in $p(i)$, making shorter parallel runs more computationally efficient than longer single runs for this problem.

---

## References

Koza, J. R. (1992). _Genetic Programming: On the Programming of Computers by Means of Natural Selection_. MIT Press, Cambridge, MA.