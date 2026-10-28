# COS 710 Assignment 3 - Structure-Based Grammatical Evolution

Predicts residential electricity load using Structure-Based GE with a domain-aware BNF grammar.

## Requirements

- Python 3.12+
- NumPy (`pip install numpy`)

## How to Run

### With Docker (recommended)

```bash
# From the GP_A3/ directory
docker build -t gp-a3 .
docker run --rm gp-a3
```

### Without Docker

```bash
pip install numpy
cd src
python3 main.py
```

Runs 10 independent evolutionary runs (seeds 42–51, 200 generations, 500 individuals each) and prints per-run results, multi-run statistics, and a comparison table against A1 and A2.

Expected runtime: ~2 minutes total.

## Project Structure

```
GP_A3/
├── Dockerfile
├── data/
│   └── Residential_Energy_Dataset_UK- 2014-2020.csv
└── src/
    ├── config.py      - all tunable parameters
    ├── grammar.py     - BNF grammar + GE derivation engine  ← key new component
    ├── tree.py        - Node and Individual data structures
    ├── primitives.py  - vectorised tree evaluation
    ├── data.py        - dataset loading and fitness-case construction
    ├── fitness.py     - fitness evaluation + tournament selection
    ├── population.py  - initial population generation
    ├── operators.py   - GE crossover, mutation, reproduction
    ├── runner.py      - main GE evolutionary loop
    ├── reporting.py   - results tables + A1/A2/A3 comparison
    └── main.py        - entry point
```

## Key Design: The BNF Grammar

The grammar encodes domain knowledge through structural templates:

```
<expr>  ::= add <expr> <expr> | sub <expr> <expr> | mul <expr> <expr> | div <expr> <expr>
          | neg <expr> | abs <expr> | sqrt <expr>
          | lag_mean <lexp> <lexp>   <- smoothed average
          | lag_max  <lexp> <lexp>   <- peak load
          | lag_min  <lexp> <lexp>   <- base load
          | lag_delta <lexp> <lexp>  <- trend
          | lag_ratio <lexp> <lexp>  <- relative change

<lexp>  ::= lag_mean <lag> <lag> | lag_max <lag> <lag> | lag_min <lag> <lag> | <lag>
<lag>   ::= x_1 | x_2 | x_3 | x_4 | x_5 | x_6 | x_7
```

Every integer chromosome is mapped to this grammar to produce an expression tree.
