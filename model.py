"""
Frequency-Weighted p-Median on Semantic Word Space
EMU414 - Mathematical Programming and Computer Applications
Hacettepe University, Spring 2025-2026

Model  (Weighted p-Median):
    minimize    sum_{w in W} sum_{v in W}  F_w * D_wv * y_wv

    s.t.
        sum_{v in W}  y_wv  =  1           for all w in W   [assignment]
        y_wv  <=  x_v                       for all w,v      [linking]
        sum_{v in W}  x_v   <=  B                            [budget]
        x_v, y_wv  in  {0,1}               for all w,v      [binary]

Interpretation:
    - x_v = 1  →  word v is selected as a representative
    - y_wv = 1  →  word w is assigned to representative v
    - Every word is assigned to exactly one representative (its nearest selected word)
    - F_w weights by frequency: frequent words penalised more if assigned far away
    - D_wv is the semantic cosine distance between words w and v

Inputs  (from data/):
    words_100.csv        columns: rank, word, freq_weight
    distance_matrix.csv  100x100 cosine-distance matrix  (index = word)

Outputs:
    model.lp             Gurobi LP export
    gurobi.log           Gurobi solver log
    results/solution.txt human-readable solution
"""

import os
import sys
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
BUDGET      = 10        # B : number of representative words to select

DATA_DIR    = "data"
RESULTS_DIR = "results"
WORDS_FILE  = os.path.join(DATA_DIR, "words_100.csv")
DIST_FILE   = os.path.join(DATA_DIR, "distance_matrix.csv")
LP_FILE     = "model.lp"
LOG_FILE    = "gurobi.log"
RESULT_FILE = os.path.join(RESULTS_DIR, "solution.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(WORDS_FILE):
        sys.exit(f"[ERROR] {WORDS_FILE} not found. Run data/prepare_data.py first.")
    if not os.path.exists(DIST_FILE):
        sys.exit(f"[ERROR] {DIST_FILE} not found. Run data/prepare_data.py first.")

    words_df = pd.read_csv(WORDS_FILE)
    words    = words_df["word"].tolist()
    weights  = words_df["freq_weight"].tolist()   # F_w

    dist_df  = pd.read_csv(DIST_FILE, index_col=0)
    dist     = dist_df.values.astype(float)       # 100x100 numpy array

    assert len(words) == dist.shape[0] == dist.shape[1], \
        "Mismatch between word list and distance matrix dimensions."

    return words, weights, dist


# ─────────────────────────────────────────────────────────────────────────────
# Gurobi model  —  Weighted p-Median
# ─────────────────────────────────────────────────────────────────────────────
def build_and_solve(words, weights, dist):
    n = len(words)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = gp.Model("pMedian_Words")
    model.setParam("LogFile", LOG_FILE)

    # ── Decision variables ──────────────────────────────────────────────────
    # x[v]    = 1  if word v is selected as representative
    # y[w, v] = 1  if word w is assigned to representative v
    x = model.addVars(n,    vtype=GRB.BINARY, name="x")
    y = model.addVars(n, n, vtype=GRB.BINARY, name="y")

    # ── Objective: minimize frequency-weighted total distance ────────────────
    #   min  Σ_w Σ_v  F_w * D_wv * y_wv
    model.setObjective(
        gp.quicksum(
            weights[w] * dist[w, v] * y[w, v]
            for w in range(n)
            for v in range(n)
        ),
        GRB.MINIMIZE
    )

    # ── Constraint 1  (Assignment) ──────────────────────────────────────────
    #   Σ_v y_wv = 1   ∀w  →  every word assigned to exactly one representative
    for w in range(n):
        model.addConstr(
            gp.quicksum(y[w, v] for v in range(n)) == 1,
            name=f"assign_{w}"
        )

    # ── Constraint 2  (Linking) ─────────────────────────────────────────────
    #   y_wv ≤ x_v   ∀w,v  →  can only assign to a selected representative
    for w in range(n):
        for v in range(n):
            model.addConstr(y[w, v] <= x[v], name=f"link_{w}_{v}")

    # ── Constraint 3  (Budget) ──────────────────────────────────────────────
    model.addConstr(
        gp.quicksum(x[v] for v in range(n)) <= BUDGET,
        name="budget"
    )

    # ── Export LP  ──────────────────────────────────────────────────────────
    model.write(LP_FILE)

    # ── Solve  ──────────────────────────────────────────────────────────────
    model.optimize()

    return model, x, y, n


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
def write_results(model, words, weights, dist, x, y, n):
    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        print(f"[WARN] Solver status: {model.Status}. No feasible solution.")
        return

    status_str = "OPTIMAL" if model.Status == GRB.OPTIMAL else "TIME LIMIT"

    # Selected representatives
    reps = [v for v in range(n) if x[v].X > 0.5]

    # Assignment: word w → representative v
    assignments = {}
    for w in range(n):
        for v in range(n):
            if y[w, v].X > 0.5:
                assignments[w] = v
                break

    # Objective breakdown
    total_weighted_dist = model.ObjVal
    max_possible_dist   = sum(weights)   # if every word assigned to distance=1 rep

    lines = [
        "=" * 65,
        "  FREQUENCY-WEIGHTED p-MEDIAN — SOLUTION",
        "=" * 65,
        f"  Status                   : {status_str}",
        f"  Objective (min weighted dist) : {total_weighted_dist:.6f}",
        f"  Budget (B)               : {BUDGET}",
        "=" * 65,
        "",
        f"SELECTED REPRESENTATIVES ({len(reps)}):",
    ]

    for v in reps:
        lines.append(f"  [{v:3d}]  {words[v]:<25}  (freq_weight = {weights[v]:.4f})")

    lines += ["", f"ASSIGNMENTS  (word  ->  representative  |  distance):"]
    for w in range(n):
        v   = assignments.get(w, -1)
        rep = words[v] if v >= 0 else "???"
        d   = dist[w, v] if v >= 0 else -1
        marker = " (self)" if w == v else ""
        lines.append(
            f"  {words[w]:<25}  ->  {rep:<25}  d={d:.4f}{marker}"
        )

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\n[INFO] Results saved to: {RESULT_FILE}")
    print(f"[INFO] LP file saved to: {LP_FILE}")
    print(f"[INFO] Log saved to:     {LOG_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity sweep  (optional, --sweep flag)
# ─────────────────────────────────────────────────────────────────────────────
def sensitivity_sweep(words, weights, dist, budgets=(5, 10, 15, 20)):
    """Run model for each budget value; report total weighted distance."""
    n = len(words)
    print("\n" + "=" * 55)
    print("  SENSITIVITY ANALYSIS  (budget sweep)")
    print("=" * 55)
    print(f"  {'B':>4}   {'Obj (min dist)':>16}   {'Avg dist/word':>14}")
    print("-" * 55)

    for B in budgets:
        m = gp.Model()
        m.setParam("OutputFlag", 0)
        x = m.addVars(n,    vtype=GRB.BINARY)
        y = m.addVars(n, n, vtype=GRB.BINARY)
        m.setObjective(
            gp.quicksum(weights[w] * dist[w, v] * y[w, v]
                        for w in range(n) for v in range(n)),
            GRB.MINIMIZE
        )
        for w in range(n):
            m.addConstr(gp.quicksum(y[w, v] for v in range(n)) == 1)
        for w in range(n):
            for v in range(n):
                m.addConstr(y[w, v] <= x[v])
        m.addConstr(gp.quicksum(x[v] for v in range(n)) <= B)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            print(f"  {B:>4}   {m.ObjVal:>16.6f}   {m.ObjVal/n:>14.6f}")
        else:
            print(f"  {B:>4}   {'N/A':>16}")

    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Frequency-Weighted p-Median on semantic word space."
    )
    parser.add_argument("--budget", type=int, default=BUDGET,
                        help=f"Budget B (default: {BUDGET})")
    parser.add_argument("--sweep", action="store_true",
                        help="Run sensitivity sweep over different budgets")
    args = parser.parse_args()

    BUDGET = args.budget

    print("[INFO] Loading data ...")
    words, weights, dist = load_data()
    print(f"[INFO] {len(words)} words loaded.")

    if args.sweep:
        sensitivity_sweep(words, weights, dist)
    else:
        print(f"[INFO] Solving p-Median  (B={BUDGET}) ...")
        model, x, y, n = build_and_solve(words, weights, dist)
        write_results(model, words, weights, dist, x, y, n)
