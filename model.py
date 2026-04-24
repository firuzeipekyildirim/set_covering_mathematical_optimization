"""
Maximum Coverage Problem on Semantic Word Space
EMU414 - Mathematical Programming and Computer Applications
Hacettepe University, Spring 2025-2026

Model:
    maximize    sum_j y[j]
    s.t.        y[j] <= sum_{i in N(j)} x[i]    for all j  (coverage)
                sum_i x[i] <= BUDGET                        (budget)
                x[i], y[j] in {0, 1}                        (binary)

where N(j) = {i : d[i][j] <= COVERAGE_RADIUS}
"""

import os
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# ---------------------------------------------------------------------------
# Parameters — edit these to change the problem instance
# ---------------------------------------------------------------------------
BUDGET = 10             # B: number of representative words to select
COVERAGE_RADIUS = 0.3   # r: cosine distance threshold for coverage

DATA_DIR = "data"
RESULTS_DIR = "results"
WORDS_FILE = os.path.join(DATA_DIR, "words_100.csv")
DISTANCE_FILE = os.path.join(DATA_DIR, "distance_matrix.csv")


def load_data():
    words_df = pd.read_csv(WORDS_FILE)
    words = words_df["word"].tolist()

    dist_df = pd.read_csv(DISTANCE_FILE, index_col=0)
    dist = dist_df.values  # numpy array, shape (100, 100)

    return words, dist


def build_neighborhoods(dist, r):
    """N[j] = list of indices i such that d[i][j] <= r"""
    n = dist.shape[0]
    neighborhoods = {}
    for j in range(n):
        neighborhoods[j] = [i for i in range(n) if dist[i][j] <= r]
    return neighborhoods


def run_model(words, dist):
    n = len(words)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    neighborhoods = build_neighborhoods(dist, COVERAGE_RADIUS)

    model = gp.Model("MaxCoverage_Words")
    model.setParam("LogFile", "gurobi.log")

    # Decision variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")  # select word i
    y = model.addVars(n, vtype=GRB.BINARY, name="y")  # word j is covered

    # Objective: maximize total coverage
    model.setObjective(gp.quicksum(y[j] for j in range(n)), GRB.MAXIMIZE)

    # Coverage constraints: y[j] can be 1 only if some x[i] in N(j) is 1
    for j in range(n):
        if neighborhoods[j]:
            model.addConstr(
                y[j] <= gp.quicksum(x[i] for i in neighborhoods[j]),
                name=f"cover_{j}"
            )
        else:
            # No word can cover j — force y[j] = 0
            model.addConstr(y[j] == 0, name=f"cover_{j}_empty")

    # Budget constraint
    model.addConstr(gp.quicksum(x[i] for i in range(n)) <= BUDGET, name="budget")

    # Export LP formulation
    model.write("model.lp")

    model.optimize()

    # Write results
    write_results(model, words, x, y, n)


def write_results(model, words, x, y, n):
    result_path = os.path.join(RESULTS_DIR, "solution.txt")
    with open(result_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MAXIMUM COVERAGE PROBLEM — SOLUTION\n")
        f.write("=" * 60 + "\n\n")

        if model.Status == GRB.OPTIMAL:
            f.write(f"Status        : OPTIMAL\n")
        else:
            f.write(f"Status        : {model.Status}\n")

        f.write(f"Objective     : {model.ObjVal:.0f} words covered\n")
        f.write(f"Budget (B)    : {BUDGET}\n")
        f.write(f"Radius (r)    : {COVERAGE_RADIUS}\n\n")

        selected = [words[i] for i in range(n) if x[i].X > 0.5]
        covered = [words[j] for j in range(n) if y[j].X > 0.5]

        f.write(f"Selected words ({len(selected)}):\n")
        for w in selected:
            f.write(f"  {w}\n")

        f.write(f"\nCovered words ({len(covered)} / {n} = "
                f"{100*len(covered)/n:.1f}%):\n")
        for w in covered:
            f.write(f"  {w}\n")

    print(f"\nResults written to {result_path}")


if __name__ == "__main__":
    words, dist = load_data()
    run_model(words, dist)
