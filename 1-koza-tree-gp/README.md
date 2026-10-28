# Genetic Programming for Electricity Load Forecasting

A GP system that evolves symbolic expressions to predict residential electricity consumption. Implements the Koza (1992) framework from scratch in Python using only NumPy.

**Course:** COS710 - Artificial Intelligence
**Dataset:** UK Residential Energy Dataset (2014–2020), 201,604 rows at 15-minute intervals

---

## Project Structure

```
GP_A1/
├── src/
│   ├── main.py             # Entry point
│   ├── config.py           # All hyperparameters
│   ├── tree.py             # Node and Individual data structures
│   ├── primitives.py       # Protected math ops + vectorised tree evaluator
│   ├── population.py       # Ramped Half-and-Half initialisation
│   ├── fitness.py          # Fitness evaluation and FPS selection
│   ├── operators.py        # Crossover, mutation, reproduction
│   ├── data.py             # CSV loading, fitness case + test set construction
│   ├── runner.py           # Single GP run
│   └── reporting.py        # Multi-run statistics and computational effort
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── data/
    └── Residential_Energy_Dataset_UK- 2014-2020.csv
```

---

## Requirements

- Python 3.9+, NumPy >= 1.24

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Local
pip install -r requirements.txt
python3 src/main.py

# Docker
docker-compose up --build
# or
docker build -t gp_a1 . && docker run --rm -v "$(pwd)/data:/app/data" gp_a1
```

---

## Configuration

All hyperparameters are in [config.py](config.py).

| Parameter | Value | Description |
|---|---|---|
| `POP_SIZE` | 500 | Individuals per generation |
| `MAX_GENERATIONS` | 250 | Max generations per run |
| `N_RUNS` | 20 | Independent runs |
| `BASE_SEED` | 42 | Run k uses seed `BASE_SEED + k` |
| `MIN_INIT_DEPTH` | 2 | Min tree depth at init |
| `MAX_INIT_DEPTH` | 8 | Max tree depth at init |
| `MAX_TREE_DEPTH` | 17 | Hard cap on depth after operators |
| `MAX_TREE_NODES` | 100 | Hard cap on node count after operators |
| `ERC_MIN / ERC_MAX` | -5.0 / 5.0 | Ephemeral Random Constant range |
| `CROSSOVER_RATE` | 0.80 | Crossover proportion |
| `MUTATION_RATE` | 0.10 | Mutation proportion |
| `REPRODUCTION_RATE` | 0.10 | Reproduction proportion |
| `N_FITNESS_CASES` | 500 | Training samples per run (fixed per run) |
| `TRAIN_RATIO` | 0.80 | Fraction used for training |
| `N_DAYS_BACK` | 7 | Lagged-day inputs (x_1 … x_7) |
| `SUCCESS_THRESHOLD` | 0.14 | Early-stop if adjusted fitness ≥ this |

---

## Design

### Representation

Trees use function nodes (`add`, `sub`, `mul`, `div`, `sqrt`, `sin`, `cos`, `neg`, `abs`) and terminal nodes (variables `x_1`–`x_7` and ERCs). Protected operations guarantee closure — no NaN/Inf.

### Input Features

Seven lagged inputs - the electricity load at the same time-of-day on each of the previous seven days:

> x₁ = load at t−96, x₂ = load at t−192, …, x₇ = load at t−672  (96 readings/day)

### Fitness (Koza 1992 chain)

| Step | Formula |
|------|---------|
| Raw | `Σ|ŷ − y|` over 500 fitness cases |
| Adjusted | `1 / (1 + raw)` → (0, 1] |
| Normalised | `adjusted_i / Σ adjusted` → sums to 1.0 |

Tree evaluation is fully **vectorised** over all 500 fitness cases via NumPy.

### Algorithm

- **Init:** Ramped Half-and-Half (depths 2–8, Full + Grow, duplicate rejection)
- **Selection:** Fitness-Proportionate Selection (roulette wheel)
- **Operators:** 80% subtree crossover, 10% subtree mutation, 10% reproduction
- **Loop:** Generational with single-individual elitism
- **Termination:** max 250 generations or adjusted fitness ≥ 0.14

---

## Results

20 independent runs, seeds 42–61, up to 250 generations, population 500.

### Training Fitness (500-case sample per run)

| Metric | Mean | Std Dev | Best |
|--------|------|---------|------|
| Raw fitness (Σ\|error\|) | 6.38515 | 0.21838 | 6.06931 |
| Adjusted fitness | 0.13552 | 0.00399 | 0.14146 |
| Successes (adj ≥ 0.14) | - | - | **4 / 20** |

### Test Set Performance (40,321 held-out rows)

| Predictor | Test MAE (kW) | vs mean baseline |
|-----------|--------------|-----------------|
| Mean predictor | 0.021826 | - |
| Persistence (x₁ only) | 0.022305 | +2.2% worse |
| GP mean (20 runs) | 0.018123 | **−17.0%** |
| GP best (run 7, seed 48) | 0.017664 | **−19.1%** |

The GP consistently outperforms both trivial baselines across all 20 runs (best run test MAE = 0.017664 kW, 12.10% of the target range).

### Best Individual (run 20, seed 61)

```lisp
(abs (abs (sqrt (mul
  (sin (sqrt (mul (abs (abs (sin x_6))) (abs (abs (sin (abs x_1)))))))
  (abs (sin (sin (abs (abs x_7)))))))))
```

Uses three lag variables - x₁ (1-day), x₆ (6-day), x₇ (7-day) - consistent with daily and weekly periods in residential demand.

### Runtime

| Metric | Value |
|--------|-------|
| Mean time per run | 14.6 s |
| Total (20 runs) | 292.2 s (~4.9 min) |
| Computational effort (Koza) | 319,000 evals at gen 22 |

---

## Reproducibility

- Each run k uses `random.seed(BASE_SEED + k)` and `numpy.random.seed(BASE_SEED + k)` - fully deterministic.
- `log_environment()` records Python/NumPy version, OS, timestamp, and PID at startup.
