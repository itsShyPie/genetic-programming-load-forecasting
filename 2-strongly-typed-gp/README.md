# COS710 Assignment 2 - Structure-Based GP for Electricity Load Forecasting

## Requirements

- Python 3.12+
- NumPy 1.26.4

```
pip install numpy==1.26.4
```

## Running

```bash
cd src
python3 main.py
```

Output: per-generation statistics, per-run summary table, multi-run aggregate
statistics, computational effort, and a side-by-side A1 vs A2 comparison.

## Docker

```bash
# Build
docker build -t cos710-a2 .

# Run
docker run --rm cos710-a2
```

## Configuration

All parameters are in `src/config.py`.  Key parameters:

| Parameter          | Default | Description                              |
|--------------------|---------|------------------------------------------|
| `N_RUNS`           | 20      | Number of independent GP runs            |
| `POP_SIZE`         | 500     | Individuals per generation               |
| `MAX_GENERATIONS`  | 250     | Maximum generations per run              |
| `CROSSOVER_RATE`   | 0.80    | Fraction of new population via crossover |
| `MUTATION_RATE`    | 0.10    | Fraction via subtree mutation            |
| `REPRODUCTION_RATE`| 0.10    | Fraction via reproduction                |
| `SUCCESS_THRESHOLD`| 0.14    | Adjusted-fitness early-stop criterion    |

## File structure

```
src/
  stgp_types.py   - STGP type system (REAL / LOAD, function signatures)
  config.py       — all tunable parameters
  tree.py         — Node and Individual classes (with node_type attribute)
  primitives.py   — protected maths + domain-typed functions (lag_mean etc.)
  population.py   — type-aware Ramped Half-and-Half initialisation
  operators.py    — type-safe crossover and mutation
  fitness.py      — Koza (1992) four-step fitness + roulette-wheel selection
  data.py         — CSV loading and fitness-case construction
  runner.py       — main GP loop (exact operator counts, not probabilistic)
  reporting.py    — aggregate statistics and A1 vs A2 comparison
  main.py         — entry point
```
