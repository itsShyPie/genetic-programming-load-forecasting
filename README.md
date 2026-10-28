# Genetic programming for electricity load forecasting

Three genetic programming systems, each built from scratch in Python with NumPy, that evolve formulas
to forecast UK household electricity load at 15-minute resolution.

Artificial Intelligence (COS710), University of Pretoria, 2026.

| Folder | Approach | Runs | Mean test MAE | Spread across runs |
|---|---|---|---|---|
| [`1-koza-tree-gp`](1-koza-tree-gp/) | Classic tree-based GP (Koza) | 20 | 0.01812 kW (12.41% of range) | ±0.00035 |
| [`2-strongly-typed-gp`](2-strongly-typed-gp/) | Structure-based, strongly-typed GP | 20 | 0.01803 kW (12.35%) | ±0.00025 |
| [`3-grammatical-evolution`](3-grammatical-evolution/) | Structure-based grammatical evolution with a domain-aware grammar | 10 | **0.01793 kW (12.28%)** | **±0.000015** |

Test error is measured on 40,321 held-out rows. For comparison, always predicting the training mean
gives 0.02183 kW, and repeating yesterday's load at the same time gives 0.02231 kW, so the evolved
models are roughly 17 to 20% better than either. Grammatical evolution was the most accurate and by far
the most consistent: all ten runs converged on essentially the same model.

Each folder is a self-contained project with its own source, Dockerfile, README and full report
(`REPORT.md`), covering the representation, fitness function, operators, experimental setup and results.

## Data

UK residential energy data from 2014 to 2020: 201,604 readings at 15-minute intervals, with columns
`utc_timestamp`, `Electricity_load`, `Residential_electricity_price`, `Residential_solar_generation`,
`Residential_wind_generation`, `Temperature` and `Relative Humidity`. The dataset was supplied for the
module and isn't included here. To run a project, put the CSV in its `data/` folder as
`Residential_Energy_Dataset_UK- 2014-2020.csv`.

## Run

With the data in place, from any of the three folders:

```bash
docker build -t gp .
docker run --rm gp
```

Or without Docker:

```bash
pip install numpy
cd src
python3 main.py
```

`1-koza-tree-gp` reads its data through a volume: use `docker compose up` there, or the plain Python
command.

## Built with

Python, NumPy, Docker
