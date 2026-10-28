# Structure-Based Grammatical Evolution for Electricity Load Forecasting
---

## 1. Description of the Dataset

### Source and Overview

The dataset used in this project is the UK Residential Energy Dataset (2014-2020), a time-series dataset of residential electricity consumption recorded at 15-minute intervals. The data is stored in a CSV file and contains 201,604 readings spanning from 31 December 2014 to 30 September 2020 - a period of approximately 5.75 years (2,100 days).

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
| Missing values      | None                                      |

Only the `Electricity_load` column is used, treating the problem as a univariate time-series regression task: predict the load at time _t_ from lagged load values at earlier times.

### Train/Test Split

The dataset is divided chronologically:

| Partition | Rows    | Date Range                          |
| --------- | ------- | ----------------------------------- |
| Training  | 161,283 | 31/12/2014 - 07/08/2019 23:30       |
| Test      | 40,321  | 07/08/2019 23:45 - 30/09/2020 23:45 |

A chronological split is used to correctly simulate real deployment conditions - train on the past and evaluate on the unseen future.

### Input Feature Construction

Seven lagged input features are constructed, each representing the electricity load at the same time-of-day _k_ days earlier (k × 96 time steps back):

> x₁ = load at t - 96, x₂ = load at t - 192, …, x₇ = load at t - 672

These features capture both the 24-hour periodic structure (sleep cycles, cooking, heating) and the 7-day periodic structure (weekday vs. weekend demand patterns).

### Fitness Cases

A random sample of 1,500 training rows is drawn once at the start of each GE run and fixed for all generations. This sample size is larger than that used in Assignments 1 and 2 (500 rows) to ensure greater diversity in the fitness signal - a direct response to feedback that "the subsampling size is likely too small to capture enough information per evaluation cycle."

---

## 2. Performance Metric

The same Koza (1992) four-step fitness chain used in Assignments 1 and 2 is applied.

| Step         | Formula                   | Direction                     |
| ------------ | ------------------------- | ----------------------------- |
| Raw          | `Σ                        | ŷ − y                         |
| Standardised | = raw (already ≥ 0)       | lower = better                |
| Adjusted     | `1 / (1 + raw)`           | higher = better, ∈ (0, 1]     |
| Normalised   | `adjusted_i / Σ adjusted` | sums to 1.0 across population |

Raw fitness is the sum of absolute errors (SAE) over all fitness cases. SAE is scale-invariant, robust to outliers, and produces values in kW - making results directly interpretable.

Important note on comparison: Because A3 uses 1,500 fitness cases and A1/A2 used 500, the raw fitness and adjusted fitness values are not directly comparable. Raw fitness is approximately 3× larger in A3 for the same per-case MAE, and adjusted fitness is correspondingly smaller. All cross-assignment comparisons use Mean Absolute Error per fitness case (MAE = raw / n_cases), which is comparable across all three assignments.

---

## 3. Structure-Based Grammatical Evolution

### What is Canonical Grammatical Evolution?

Grammatical Evolution (GE), introduced by Ryan, Collins, and Neill (1998), is an evolutionary algorithm in which individuals are represented as integer arrays (chromosomes) rather than expression trees. A BNF (Backus-Naur Form) grammar maps each chromosome to a program (the phenotype) via the following derivation process:

1. The grammar defines a set of non-terminal symbols and production rules.
2. The derivation begins with the start symbol.
3. For each non-terminal encountered (left-to-right), the next integer codon in the chromosome is consumed; the production rule is selected as `codon % number_of_alternatives`.
4. If all codons are consumed before derivation is complete, the chromosome _wraps_: the pointer resets to the start and reuses codons (up to `MAX_WRAPS` times).
5. If derivation fails (too many wraps, or depth limit reached with no valid continuation), the individual is declared invalid and receives maximum-penalty fitness.

This genotype-phenotype separation is the defining feature of GE: the chromosome evolves in integer space, and the grammar defines which programs are reachable.

### How Structure is Incorporated into Canonical GE

Structure is incorporated into GE by designing the grammar itself to encode domain knowledge. This is fundamentally different from both canonical GP (A1, which uses an unrestricted function set) and Strongly Typed GP (A2, which restricts type-compatible subtree swaps but still builds arbitrary trees).

In Structure-Based GE, the grammar defines structural templates - production rules that directly represent physically meaningful operations on electricity load data. These templates are first-class citizens of the grammar: they are alternative productions for the `<expr>` non-terminal, alongside standard arithmetic. The grammar also introduces a dedicated `<lexp>` ("load expression") non-terminal that constrains what arguments structural templates may accept, enabling controlled nesting.

The structural nature of the approach means that:
- Every individual is guaranteed to use at least one domain-meaningful building block (a structural template), enforced by the grammar's depth-limit fallback which always selects a structural template.
- The grammar channels the search toward expressions that domain experts would recognise as physically sensible.
- This guidance is provided without explicit type annotations on nodes (as in STGP) - instead, the grammar defines the valid structural patterns directly.

---

## 4. Description of the Grammar

The grammar uses prefix notation (Polish/S-expression order): the function name appears first, followed by its arguments as non-terminal symbols. This maps directly to the expression tree used in evaluation.

```
<expr>   ::=  add  <expr> <expr>        (binary addition)
           |  sub  <expr> <expr>        (binary subtraction)
           |  mul  <expr> <expr>        (binary multiplication)
           |  div  <expr> <expr>        (protected division)
           |  neg  <expr>               (unary negation)
           |  abs  <expr>               (unary absolute value)
           |  sqrt <expr>               (protected square root)
           |  lag_mean  <lexp> <lexp>   (STRUCTURAL: smoothed average)
           |  lag_max   <lexp> <lexp>   (STRUCTURAL: peak load estimate)
           |  lag_min   <lexp> <lexp>   (STRUCTURAL: base load estimate)
           |  lag_delta <lexp> <lexp>   (STRUCTURAL: day-over-day trend)
           |  lag_ratio <lexp> <lexp>   (STRUCTURAL: relative change)

<lexp>   ::=  lag_mean <lag> <lag>      (nested average of two raw lags)
           |  lag_max  <lag> <lag>      (nested max of two raw lags)
           |  lag_min  <lag> <lag>      (nested min of two raw lags)
           |  <lag>                     (single lag variable)

<lag>    ::=  x_1 | x_2 | x_3 | x_4 | x_5 | x_6 | x_7
```

### Grammar Non-Terminal Counts

| Non-terminal | Number of alternatives | Codon mapping |
| ------------ | ---------------------- | ------------- |
| `<expr>`     | 12                     | `codon % 12`  |
| `<lexp>`     | 4                      | `codon % 4`   |
| `<lag>`      | 7                      | `codon % 7`   |

### Design Decisions

Structural templates as `<expr>` alternatives. The five structural templates appear alongside binary arithmetic as alternatives for any `<expr>`. This gives the search the freedom to combine templates with arithmetic - for example, `add(lag_mean(x₁, x₇), lag_delta(x₃, x₅))` mixes a baseline average with a trend signal.**

The `<lexp>` non-terminal. Structural templates take `<lexp>` arguments rather than `<expr>` arguments. This allows one level of nesting within a structural template: `lag_mean(lag_mean(x₁, x₇), x₄)` is valid (the inner `lag_mean(x₁, x₇)` satisfies `<lexp>`), but arbitrary arithmetic cannot appear inside a structural template's arguments. This mirrors the LOAD type system from A2 but is enforced through grammar rules rather than explicit type annotations.

Depth-limit fallback to structural templates. When the derivation depth limit (`MAX_DEPTH = 8`) is reached, the grammar forces selection from the five structural templates (rules 7-11 for `<expr>`). This guarantees that even deep recursive trees always terminate with a domain-meaningful building block.

No ephemeral random constants (ERCs). The grammar omits ERCs. The structural templates operate on load readings (bounded in [0.031, 0.177] kW) and their combinations - scaling factors are unnecessary. As confirmed by A2's best individual (which contained no ERCs), the optimal expression for this problem is a weighted average of raw lag values, not a scaled arithmetic expression.

---

## 5. Initial Population Generation

The initial population consists of `POP_SIZE = 500` individuals, each with a randomly generated chromosome.

**Chromosome generation.** Each chromosome is an integer array of length `CHROM_LEN = 30`. Each codon is drawn uniformly from `[0, 255]` using the run's random seed.

Why CHROM_LEN = 30? The active portion of a chromosome (the codons actually consumed during derivation) ranges from approximately 5 codons (for a simple `lag_mean(x_i, x_j)`) to 15 codons (for a complex expression like `add(lag_mean(lag_mean(x_1, x_7), x_4), lag_delta(x_3, x_6))`). With `CHROM_LEN = 30`, the intron fraction (unused codons) is kept around 50-70%, ensuring that single-point crossover has a reasonable probability of hitting an active codon and producing meaningful variation. A longer chromosome (128 was initially tried) led to a >95% intron fraction, rendering crossover effectively random.

**Tree derivation at initialization.** Immediately after creating each chromosome, the grammar derivation is run to produce the phenotype (expression tree). Invalid individuals (derivation failed due to exhausted wraps) are kept in the population and receive penalty fitness during evaluation. In practice, with `CHROM_LEN = 30` and `MAX_WRAPS = 3`, approximately 8-10% of initial individuals are invalid (the rest 90-92% are valid).

---

## 6. Fitness Function and Fitness Evaluation

### Fitness Function

The raw fitness of an individual is the sum of absolute errors (SAE) over the 1,500 fitness cases:

$$r(i) = \sum_{j=1}^{1500} \left| \hat{y}_j - y_j \right|$$

where $\hat{y}_j$ is the prediction for fitness case $j$ and $y_j$ is the actual electricity load. Adjusted fitness is:

$$a(i) = \frac{1}{1 + r(i)}$$

### Handling Invalid Individuals

Invalid individuals (where the grammar derivation failed to produce a complete expression tree) receive a penalty raw fitness of $\text{PENALTY} = 1{,}500 \times 10 = 15{,}000$, which maps to an adjusted fitness of approximately $6.7 \times 10^{-5}$ - far below any valid individual. This effectively removes invalid individuals from selection while avoiding the cost of regenerating them.

### Evaluation

The expression tree is evaluated over all 1,500 fitness cases simultaneously using NumPy array operations (vectorised evaluation), as in Assignments 1 and 2. Protected operations prevent runtime errors:
- Protected division (`div`): returns 1.0 when |denominator| < 10⁻¹⁰.
- Protected square root (`sqrt`): computes √|a|.
- Overflow protection: NaN and ±∞ values are replaced by ±10⁶ after each operation.

---

## 7. Selection Method

Tournament Selection is used, replacing the fitness-proportionate selection (roulette wheel) of Assignments 1 and 2.

For each parent required, `TOURNAMENT_SIZE = 4` individuals are sampled uniformly at random from the population (without replacement). The individual with the highest adjusted fitness among the four contestants wins the tournament and is selected as a parent.

Why tournament selection?
- Tournament selection maintains selection pressure even when adjusted fitness values are clustered (e.g., when many individuals are structurally similar or invalid).
- It is scale-independent: unlike roulette-wheel selection, the winning probability depends only on the relative rank of individuals within a tournament, not on the absolute magnitude of adjusted fitness values.
- A tournament size of 4 provides moderate selection pressure - lower than the k=7 initially tried (which caused the entire population to converge to a single phenotype within 10 generations) but sufficient to consistently select the fittest individuals.

---

## 8. Genetic Operators

Operators are applied in exact counts to fix the issue from Assignment 1:

| Operator      | Rate | Exact count (POP=500) |
| ------------- | ---- | --------------------- |
| Crossover     | 80%  | 400 offspring         |
| Mutation      | 15%  | 75 offspring          |
| Reproduction  | 5%   | 25 offspring          |

All three counts are computed before the loop begins: `n_crossover = round(500 × 0.80) = 400`, `n_mutation = round(500 × 0.15) = 75`, `n_reproduction = 500 − 400 − 75 = 25`. This guarantees that exactly the specified fraction of each generation is produced by each operator.

**Elitism:** The single best individual from the current generation is always copied unchanged into the next generation. This ensures the best solution found is never lost.

### 8.1 Crossover (Single-Point)

1. Select two parent chromosomes P₁ and P₂ via tournament selection.
2. Pick a random cut point uniformly in `[1, CHROM_LEN − 1]`.
3. Offspring chromosome = P₁[:cut] + P₂[cut:].
4. Re-derive the expression tree from the new chromosome via the grammar.

Single-point crossover swaps the tail of P₁ with the tail of P₂. If the cut falls in the active region of both chromosomes (the codons that drive the derivation), the offspring can have a structurally different phenotype from either parent. With `CHROM_LEN = 30` and active regions of 5-15 codons, a random cut point has a 17-50% probability of landing in the active region, making crossover a meaningful operator.

### 8.2 Mutation (Per-Codon Random Reset)

1. Select one parent via tournament selection.
2. Copy the chromosome.
3. For each of the 30 codons, with probability `CODON_MUTATION_PROB = 0.05`, replace the codon with a new value drawn uniformly from `[0, 255]`.
4. Re-derive the expression tree.

With 30 codons and rate 0.05, the expected number of mutated codons per individual is 1.5. If a mutated codon falls in the active region, it changes the production rule at that derivation step - potentially altering the function chosen, the lag variable selected, or whether a structural template or arithmetic operator is used.

### 8.3 Reproduction

A tournament-selected parent is copied unchanged (chromosome and derived tree) into the next generation. This preserves fit individuals across generations alongside the elitist copy of the best.

---

## 9. Experimental Setup

### Parameter Values

| Parameter             | Value   | Description                                    |
| --------------------- | ------- | ---------------------------------------------- |
| `N_DAYS_BACK`         | 7       | Lag features x₁ … x₇                          |
| `TRAIN_RATIO`         | 0.80    | First 80% of data for training                 |
| `N_FITNESS_CASES`     | 1,500   | Training rows sampled per run                  |
| `POP_SIZE`            | 500     | Individuals per generation                     |
| `CHROM_LEN`           | 30      | Integer codons per chromosome                  |
| `CODON_MAX`           | 255     | Codon range [0, 255]                           |
| `MAX_WRAPS`           | 3       | Maximum chromosome wraps during derivation     |
| `MAX_DEPTH`           | 8       | Maximum recursion depth for derivation         |
| `CROSSOVER_RATE`      | 0.80    | Fraction of offspring via crossover            |
| `MUTATION_RATE`       | 0.15    | Fraction of offspring via mutation             |
| `REPRODUCTION_RATE`   | 0.05    | Fraction of offspring via reproduction         |
| `CODON_MUTATION_PROB` | 0.05    | Per-codon mutation probability                 |
| `TOURNAMENT_SIZE`     | 4       | Tournament size for parent selection           |
| `MAX_GENERATIONS`     | 200     | Generations per run (no early exit)            |
| `N_RUNS`              | 10      | Independent runs                               |
| `BASE_SEED`           | 42      | Run k uses seed BASE_SEED + k                  |

### Parameter Justification

- `N_FITNESS_CASES = 1,500`: Increased from 500 (A1/A2) to provide a more representative sample as recommended in the A2 feedback.
- `CHROM_LEN = 30`: Short enough to keep the intron fraction low, ensuring crossover is meaningful. The alternative of 128 (initially used) produced 95%+ introns and effectively no evolution.
- `TOURNAMENT_SIZE = 4`: Provides moderate selection pressure without causing immediate premature convergence.
- `MAX_GENERATIONS = 200` with no early-exit threshold: Directly addresses A2 feedback that "termination criteria is likely stopping the run way too early." All 200 generations are always executed.

### Technical Specifications

| Component          | Value                                                |
| ------------------ | ---------------------------------------------------- |
| Python             | 3.12.3                                               |
| NumPy              | 1.26.4                                               |
| OS                 | Linux WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2) |
| External libraries | NumPy only                                           |
| GE implementation  | From scratch                                         |

---

## 10. Results and Discussion

### Per-Run Summary

Ten independent runs were performed (seeds 42-51, 200 generations each):

| Run | Seed | Final Adj | Final Raw | Train MAE (kW) | Time (s) | Test MAE (kW) |
| --- | ---- | --------- | --------- | -------------- | -------- | ------------- |
| 1   | 42   | 0.05121   | 18.5260   | 0.012351       | 13.6     | 0.017935      |
| 2   | 43   | 0.04889   | 19.4532   | 0.012969       | 12.7     | 0.017935      |
| 3   | 44   | 0.04725   | 20.1655   | 0.013444       | 12.4     | 0.017935      |
| 4   | 45   | 0.04925   | 19.3057   | 0.012870       | 11.7     | 0.017935      |
| 5   | 46   | 0.05021   | 18.9147   | 0.012610       | 11.8     | 0.017898      |
| 6   | 47   | 0.04823   | 19.7325   | 0.013155       | 11.8     | 0.017898      |
| 7   | 48   | 0.04855   | 19.5958   | 0.013064       | 11.9     | 0.017929      |
| 8   | 49   | 0.05021   | 18.9180   | 0.012612       | 12.6     | 0.017929      |
| 9   | 50   | 0.05065   | 18.7428   | 0.012495       | 11.9     | 0.017935      |
| 10  | 51   | 0.04841   | 19.6575   | 0.013105       | 11.8     | 0.017935      |

### Multi-Run Summary

| Metric                              | Mean     | Std Dev  | Best     |
| ----------------------------------- | -------- | -------- | -------- |
| Raw fitness (Σ\|error\|, 1500 cases) | 19.3012  | 0.5141   | 18.5260  |
| Train MAE per case (kW)            | 0.012867 | 0.000343 | 0.012351 |
| Train MAE as % of target range     | 8.81%    | 0.23%    | 8.46%    |
| Adjusted fitness 1/(1+raw)         | 0.04929  | 0.00125  | 0.05121  |
| Test MAE (kW, 40,321 rows)         | 0.017926 | 0.000015 | 0.017898 |
| Test MAE as % of target range      | 12.28%   | 0.01%    | 12.26%   |

**Runs: 10 | Mean time per run: 12.2 s | Total time: 122.1 s (2.0 min)**

### Best Individual

The best individual across all runs was found in run 1 (seed 42) with a training MAE of 0.012351 kW and a test MAE of 0.017935 kW:

```lisp
(lag_mean (lag_mean x_7 x_4) (lag_mean x_6 x_1))
```

This expression is algebraically equivalent to:

$$\hat{y}(t) = \frac{1}{4}(x_1 + x_4 + x_6 + x_7)$$

which is an unweighted average of the load from 1, 4, 6, and 7 days ago at the same time-of-day. This is the **same expression** as Assignment 2's best individual (`lag_mean(lag_mean(x_4, x_7), lag_mean(x_6, x_1))`), confirming that the GE grammar successfully guides the search to the same optimal structural pattern.

Key observations:
- **x₇ (7-day lag)** consistently appears in the best individuals, confirming the dominance of the weekly periodicity in residential demand.
- **No arithmetic operators** appear in the best individuals - the optimal prediction is a pure structural combination of lag values, with no scaling.
- The grammar's `<lexp>` non-terminal directly enabled this nested averaging structure: `lag_mean(lag_mean(x₁, x₇), ...)` is a valid derivation from `lag_mean <lexp> <lexp>` where the first `<lexp>` expands to `lag_mean x₁ x₇`.

### Evolutionary Dynamics

Each run improves from the random initial population over approximately 10 generations before converging:

- **Generation 0:** Approximately 8-10% of individuals are invalid (derivation failed). The best individual has adj ≈ 0.047-0.049.
- **Generation 10:** The population has converged to the best structural pattern (4-lag average or similar). The best individual improves to adj ≈ 0.048-0.051.
- **Generations 10-200:** The best fitness remains stable. The population has converged to the optimal neighborhood in phenotype space.

This rapid convergence reflects the strong structural bias of the grammar: the `<lexp>` non-terminal ensures that all individuals combine lag variables via domain-meaningful averaging or comparison operations. The search space of meaningful structural patterns is small (essentially variations of k-lag averages over different subsets of {x₁, …, x₇}), and the optimal pattern (4-lag average including x₁ and x₇) is found within the first 10 generations.

While the fitness plateau after generation 10 might appear to indicate insufficient evolution, the convergence to a globally strong solution (the lowest mean test MAE of all three assignments) demonstrates that the structural grammar successfully guided the search to the optimal neighborhood rapidly, rather than wandering through a larger, less structured search space as in A1 and A2. The plateau is not a failure of the evolutionary process - it is evidence that the grammar has compressed the search space to the point where the optimal region is reached quickly and held thereafter.

### A3-Specific Insights

Why convergence is so rapid. The grammar constrains every individual to use structural templates (`lag_mean`, `lag_max`, `lag_min`, `lag_delta`, `lag_ratio`) as either the root expression or as components within arithmetic combinations. Within this constrained space, the best-performing family of expressions - unweighted averages of 3-4 lag variables - is easily reachable from almost any random starting chromosome. Because the `<lexp>` non-terminal limits structural template arguments to raw lags or simple nested templates, the number of distinct high-fitness phenotypes is small, and random initialisation produces at least some near-optimal individuals in the very first population. Crossover and mutation then quickly consolidate the population around these patterns.

Why all runs find the same structural pattern. Every run converges to a variant of `lag_mean(lag_mean(xᵢ, x₇), lag_mean(xⱼ, x₁))` - a 4-lag average that always includes x₁ (yesterday) and x₇ (same weekday last week). The consistency of x₁ and x₇ across runs reflects the underlying structure of the data: residential electricity consumption is dominated by a 24-hour cycle (captured by x₁) and a 7-day cycle (captured by x₇). The grammar's structural templates allow the search to discover these dominant patterns directly, without needing to decompose them from raw arithmetic as A1 was forced to do.

The role of invalid individuals. At generation 0, approximately 8-10% of chromosomes produce invalid derivations (the grammar fails to terminate within MAX_WRAPS = 3 wraps). These receive penalty fitness and are strongly disfavoured by tournament selection. By generation 10, invalids have effectively been eliminated from the population - the valid fraction reaches 99-100% and stays there. This self-correction happens naturally: crossover and mutation between valid parents tend to produce valid offspring because the chromosome lengths (typically 5-15 active codons) are well within the 30-codon budget.

Generalisation gap. The mean train MAE (0.012867 kW, measured on 1,500 sampled rows) is lower than the mean test MAE (0.017926 kW, measured on 40,321 rows). This train-test ratio of approximately 1.39× is consistent with A1 (1.42×) and A2 (1.44×). The gap arises from two sources: (1) the 1,500-case sample does not perfectly represent the full training distribution, so the expression is slightly over-fitted to the sample; (2) the test period (August 2019 - September 2020) includes the COVID-19 lockdown period (March-September 2020), during which residential electricity consumption patterns changed markedly relative to the training years. Since the training data (up to August 2019) contains no lockdown observations, the evolved predictors cannot account for this distributional shift.

What the grammar prevents. The grammar's design also reveals what was unhelpful in A1: deeply nested arithmetic involving trigonometric functions (`sin`, `cos`) and scaling constants. These functions appeared in A1's best individuals but did not improve accuracy - the fitness landscape rewards simple lag averages, and the extra arithmetic just added noise. A3's grammar makes such expressions unreachable (the grammar has no trigonometric productions), and this restriction does not hurt performance: A3 achieves better mean test MAE than A1 despite a strictly smaller function set.

### Variance and Consistency

The standard deviation of test MAE across runs is 0.000015 kW - remarkably low, less than 0.1% of the mean. This extreme consistency indicates that all 10 runs converge to essentially the same family of expressions (4-lag averages of different lag subsets that achieve near-identical test MAE). The grammar-guided search is highly reproducible.

---

## 11. Runtimes

| Statistic             | Value     |
| --------------------- | --------- |
| Mean time per run     | 12.2 s    |
| Total time (10 runs)  | 122.1 s   |
| Approx. time per generation | 61 ms |

The 12.2 s/run runtime sits between A1 (14.6 s/run) and A2 (41.6 s/run) in absolute terms, but uses 1,500 fitness cases vs. 500 in A1/A2. Normalised to 500 cases, A3 would take approximately 4.1 s/run - faster than A1 because GE chromosome derivation is simpler than tree traversal for type-safe crossover/mutation.

---

## 12. Comparison with Assignments 1 and 2

### Three-Way Summary Table

| Metric                     | A1 (Canonical GP) | A2 (STGP) | A3 (GE)      |
| -------------------------- | ----------------- | --------- | ------------ |
| Mean Train MAE (kW)        | 0.012770          | 0.012522  | 0.012867     |
| Best Train MAE (kW)        | 0.012139          | 0.011939  | 0.012351     |
| Mean Test MAE (kW)         | 0.018123          | 0.018027  | **0.017926** |
| Best Test MAE (kW)         | 0.017664          | 0.017672  | 0.017898     |
| Mean Test MAE (% of range) | 12.41%            | 12.35%    | **12.28%**   |
| Test MAE Std Dev (kW)      | 0.000350          | 0.000251  | **0.000015** |
| N runs                     | 20                | 20        | 10           |
| Mean time per run (s)      | 14.6              | 41.6      | 12.2         |

### Discussion

#### Accuracy

A3 achieves the lowest mean test MAE of all three assignments: 0.017926 kW vs. 0.018027 kW (A2) and 0.018123 kW (A1). This represents a 1.1% improvement over A1. The grammar-based structural bias guided the search directly toward 4-lag averaging expressions, which are the best predictors for this weekly-periodic dataset.

However, A3's best test MAE (0.017898 kW) is slightly worse than A1 (0.017664 kW) and A2 (0.017672 kW). This is because A1 and A2 were run for 20 runs each, giving more opportunities to find a lucky seed that finds a slightly better expression. With 10 runs, A3's best is slightly higher. A direct comparison requires equal numbers of runs.

#### Consistency

The most striking result is A3's **standard deviation of 0.000015 kW** - approximately 17× lower than A2 (0.000251) and 23× lower than A1 (0.000350). This means that A3 is highly reliable: virtually every run produces the same near-optimal prediction formula. A2 showed improved consistency over A1 through its type system; A3's grammar-based structure achieves a further order-of-magnitude improvement in consistency.

This reliability is particularly valuable in practice: a user running A3 just once can be confident of obtaining a near-optimal predictor.

#### Interpretability

All three assignments' best individuals are interpretable, but they differ in complexity:
- A1 best: A complex tree combining `abs`, `sin`, `sqrt`, and multiple lags - algebraically analysable but not immediately domain-readable.
- A2 best: `lag_mean(lag_mean(x₄, x₇), lag_mean(x₆, x₁))` - a 4-point moving average; simple and interpretable.
- A3 best: `lag_mean(lag_mean(x₇, x₄), lag_mean(x₆, x₁))` - identical structure to A2's best (same lags, different ordering within symmetric function).

A3's grammar guaranteed that the search found an interpretable, domain-meaningful expression. The grammar itself communicates what the model can do (combine historical loads in meaningful ways) and the evolved expression is what any domain expert would have considered first.

#### Speed

A3 (12.2 s/run) is faster than A2 (41.6 s/run), even though A3 uses 3× more fitness cases. GE crossover and mutation operate on flat integer arrays in O(CHROM_LEN) time - much faster than A2's tree traversal for type-compatible crossover point search, which requires scanning the entire tree of each parent. A3 is also slightly faster than A1 (14.6 s/run, 500 cases) per fitness case.

#### Why Structure-Based GE Outperforms A1 and A2 in Mean Accuracy

The grammar restricts the search to expressions that use structural templates at every level - specifically combinations of lag variables via averaging, peak detection, and difference operations. Because the optimal predictor for this dataset IS such a combination (a 4-lag average), A3's grammar essentially defines a compressed search space that contains the optimal solution. A1 and A2 search a much larger space (any arithmetic expression), which includes the optimal solution but also contains many sub-optimal expressions that may be discovered instead.

#### Fitness Landscape

The consistency of A3's results (extremely low test MAE variance) reflects a fitness landscape where all good solutions are structurally similar: 4-lag averages over different subsets of {x₁, …, x₇}. The grammar defines this landscape directly, concentrating the search in the productive region.

In contrast, A1 and A2 search a much larger, rougher landscape where arithmetic combinations of lags, constants, and trigonometric functions can produce many locally competitive but globally sub-optimal solutions. This explains why A1 and A2 show higher run-to-run variance: different seeds reach different local optima. A3's grammar removes most of those optima from the search space, leaving a smooth, easily navigable landscape centred on the family of k-lag averages.

#### Overfitting and Generalisation

All three assignments show a comparable train-to-test MAE ratio (A1: 1.42×, A2: 1.44×, A3: 1.39×), indicating similar levels of generalisation. A3's ratio is the smallest, suggesting that the grammar's restriction to averaging-based structural templates acts as implicit regularisation: the simpler, more constrained expressions tend to generalise better than the complex arithmetic trees found by A1. The COVID-19 lockdown period in the test set (March-September 2020) introduces distributional shift that all three approaches are equally unable to handle, as none of the training data reflects lockdown-altered consumption behaviour.

---

## References

- Ryan, C., Collins, J. J., & Neill, M. O. (1998). Grammatical evolution: Evolving programs for an arbitrary language. In *Proceedings of EuroGP*, pp. 83–96.
- Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press.
