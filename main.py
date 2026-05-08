"""
Frequency-Weighted p-Median on Semantic Word Space
EMU414 - Mathematical Programming and Computer Applications
Hacettepe University, Spring 2025-2026

Model (Weighted p-Median):
    minimize    sum_{w in W} sum_{v in W}  F_w * D_wv * y_wv

    s.t.
        sum_{v in W}  y_wv  =  1           for all w in W   [assignment]
        y_wv  <=  x_v                       for all w,v      [linking]
        sum_{v in W}  x_v   <=  B                            [budget]
        x_v, y_wv  in  {0,1}               for all w,v      [binary]

Inputs:
    data/input.xlsx
        Sheet "Words"      : columns rank, word  (frequency-sorted)
        Sheet "Parameters" : columns Parameter, Value
                             rows: Budget, MIPGap, TimeLimit
    data/dolma_300_2024_1.2M.100_combined.txt
        GloVe-format 300-dim embeddings

Outputs:
    model.lp              Gurobi LP export
    gurobi.log            Gurobi solver log
    results/solution.txt  human-readable solution
"""

import os
import sys
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_FILE  = os.path.join("data", "input.xlsx")
EMBED_FILE  = os.path.join("data", "dolma_300_2024_1.2M.100_combined.txt")
RESULTS_DIR = "results"
LP_FILE     = "model.lp"
LOG_FILE    = "gurobi.log"
RESULT_FILE = os.path.join(RESULTS_DIR, "solution.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_excel(path):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] {path} not found.")

    words_df = pd.read_excel(path, sheet_name="Words")
    if not {"rank", "word"}.issubset(words_df.columns):
        sys.exit("[ERROR] Sheet 'Words' must have columns: rank, word")

    params_df = pd.read_excel(path, sheet_name="Parameters")
    if not {"Parameter", "Value"}.issubset(params_df.columns):
        sys.exit("[ERROR] Sheet 'Parameters' must have columns: Parameter, Value")

    params = dict(zip(params_df["Parameter"].str.strip(), params_df["Value"]))
    for key in ("Budget", "MIPGap", "TimeLimit"):
        if key not in params:
            sys.exit(f"[ERROR] Sheet 'Parameters' missing row: {key}")

    words   = words_df["word"].astype(str).tolist()
    ranks   = words_df["rank"].astype(float).tolist()
    budget  = int(params["Budget"])
    mipgap  = float(params["MIPGap"])
    tlimit  = float(params["TimeLimit"])

    weights = compute_freq_weights(ranks)

    xl      = pd.ExcelFile(path)
    if "KnownWords" in xl.sheet_names:
        kw_df       = pd.read_excel(path, sheet_name="KnownWords")
        known_words = kw_df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        known_words = []

    return words, weights, budget, mipgap, tlimit, known_words


def compute_freq_weights(ranks):
    """Compute normalised frequency weights from rank list. Modify here to change weighting scheme."""
    raw   = [1.0 / r for r in ranks]
    max_w = max(raw)
    return [(w / max_w) ** 1.1 for w in raw]


def load_embeddings(path, words):
    if not os.path.exists(path):
        sys.exit(
            f"[ERROR] {path} not found.\n"
            f"        Download embeddings from:\n"
            f"        https://nlp.stanford.edu/data/wordvecs/glove.2024.dolma.300d.zip\n"
            f"        Extract the .txt file into the data/ directory."
        )

    needed  = set(words)
    found   = {}

    print(f"[INFO] Scanning embeddings for {len(needed)} words ...")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            token = parts[0]
            if token in needed:
                found[token] = np.array(parts[1:], dtype=np.float32)
                if len(found) == len(needed):
                    break

    missing = needed - set(found)
    if missing:
        print(f"[WARN] {len(missing)} words not found in embeddings: {missing}")

    dim = next(iter(found.values())).shape[0] if found else 0
    print(f"[INFO] Loaded {len(found)} embeddings (dim={dim}).")
    return found


def build_distance_matrix(words, embeddings):
    n    = len(words)
    vecs = []
    for w in words:
        if w in embeddings:
            vecs.append(embeddings[w])
        else:
            vecs.append(np.zeros(next(iter(embeddings.values())).shape[0], dtype=np.float32))
    V = np.stack(vecs)   # (n, dim)

    # L2 distance: ||v_i - v_j||_2
    diff = V[:, None, :] - V[None, :, :]   # (n, n, dim)
    dist = np.sqrt((diff ** 2).sum(axis=-1))   # (n, n)
    return dist


# ─────────────────────────────────────────────────────────────────────────────
# Gurobi model  —  Weighted p-Median
# ─────────────────────────────────────────────────────────────────────────────
def build_and_solve(words, weights, dist, budget, mipgap, tlimit, known_indices):
    n = len(words)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = gp.Model("pMedian_Words")
    model.setParam("LogFile", LOG_FILE)
    model.setParam("MIPGap",  mipgap)
    model.setParam("TimeLimit", tlimit)

    x = model.addVars(n,    vtype=GRB.BINARY, name="x")
    y = model.addVars(n, n, vtype=GRB.BINARY, name="y")

    # Force known words to be selected as representatives
    for v in known_indices:
        model.addConstr(x[v] == 1, name=f"known_{v}")

    model.setObjective(
        gp.quicksum(
            weights[w] * dist[w, v] * y[w, v]
            for w in range(n)
            for v in range(n)
        ),
        GRB.MINIMIZE
    )

    for w in range(n):
        model.addConstr(
            gp.quicksum(y[w, v] for v in range(n)) == 1,
            name=f"assign_{w}"
        )

    for w in range(n):
        for v in range(n):
            model.addConstr(y[w, v] <= x[v], name=f"link_{w}_{v}")

    # Known words do not count toward budget
    free_indices = [v for v in range(n) if v not in known_indices]
    model.addConstr(
        gp.quicksum(x[v] for v in free_indices) <= budget,
        name="budget"
    )

    model.write(LP_FILE)
    model.optimize()

    return model, x, y, n


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
def write_results(model, words, weights, dist, x, y, n, budget, input_file):
    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        print(f"[WARN] Solver status: {model.Status}. No feasible solution.")
        return

    status_str = "OPTIMAL" if model.Status == GRB.OPTIMAL else "TIME LIMIT"

    reps = [v for v in range(n) if x[v].X > 0.5]

    assignments = {}
    for w in range(n):
        for v in range(n):
            if y[w, v].X > 0.5:
                assignments[w] = v
                break

    total_weighted_dist = model.ObjVal

    summary = [
        "=" * 65,
        "  FREQUENCY-WEIGHTED p-MEDIAN — SOLUTION",
        "=" * 65,
        f"  Status                       : {status_str}",
        f"  Objective (min weighted dist) : {total_weighted_dist:.6f}",
        f"  Budget (B)                    : {budget}",
        "=" * 65,
        "",
        f"SELECTED REPRESENTATIVES ({len(reps)}):",
    ]
    for v in reps:
        summary.append(f"  [{v:3d}]  {words[v]:<25}  (freq_weight = {weights[v]:.4f})")

    assignment_lines = ["", "ASSIGNMENTS  (word  ->  representative  |  L2 distance):"]
    for w in range(n):
        v      = assignments.get(w, -1)
        rep    = words[v] if v >= 0 else "???"
        d      = dist[w, v] if v >= 0 else -1.0
        marker = " (self)" if w == v else ""
        assignment_lines.append(
            f"  {words[w]:<25}  ->  {rep:<25}  d={d:.4f}{marker}"
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(summary + assignment_lines))

    print("\n".join(summary))

    solution_path = write_solution_excel(
        model, words, weights, dist, reps, assignments, n, budget,
        status_str, total_weighted_dist, input_file
    )

    print(f"\n[INFO] Solution Excel  : {solution_path}")
    print(f"[INFO] Results txt     : {RESULT_FILE}")
    print(f"[INFO] LP file         : {LP_FILE}")
    print(f"[INFO] Gurobi log      : {LOG_FILE}")


def write_solution_excel(model, words, weights, dist, reps, assignments,
                         n, budget, status_str, obj_val, input_file):
    import openpyxl
    from openpyxl import load_workbook

    out_path = os.path.splitext(input_file)[0] + "_solution.xlsx"

    wb = load_workbook(input_file)

    # ── Sheet: Representatives ───────────────────────────────────────────────
    ws_reps = wb.create_sheet("Representatives")
    ws_reps.append(["Index", "Word", "Freq_Weight"])
    for v in reps:
        ws_reps.append([v, words[v], round(weights[v], 6)])

    # ── Sheet: Assignments ───────────────────────────────────────────────────
    ws_asgn = wb.create_sheet("Assignments")
    ws_asgn.append(["Word", "Freq_Weight", "Representative", "L2_Distance", "Is_Self"])
    for w in range(n):
        v    = assignments.get(w, -1)
        rep  = words[v] if v >= 0 else ""
        d    = round(float(dist[w, v]), 6) if v >= 0 else ""
        self = (w == v)
        ws_asgn.append([words[w], round(weights[w], 6), rep, d, self])

    # ── Sheet: Solver ────────────────────────────────────────────────────────
    ws_solv = wb.create_sheet("Solver")
    ws_solv.append(["Parameter", "Value"])
    ws_solv.append(["Status",      status_str])
    ws_solv.append(["Objective",   round(obj_val, 8)])
    ws_solv.append(["SolveTime_s", round(model.Runtime, 4)])
    ws_solv.append(["MIPGap",      round(model.MIPGap, 8)])
    ws_solv.append(["NodeCount",   int(model.NodeCount)])
    ws_solv.append(["SolCount",    int(model.SolCount)])
    ws_solv.append(["Budget",      budget])

    wb.save(out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    args       = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags      = {a for a in sys.argv[1:] if a.startswith("--")}
    input_file = args[0] if args else EXCEL_FILE
    uniform    = "--uniform" in flags

    print("[INFO] Loading Excel ...")
    words, weights, budget, mipgap, tlimit, known_words = load_excel(input_file)
    if uniform:
        weights = [1.0] * len(words)
        print("[INFO] --uniform flag set: all freq_weights = 1")

    word_index    = {w: i for i, w in enumerate(words)}
    known_indices = set()
    for kw in known_words:
        if kw in word_index:
            known_indices.add(word_index[kw])
        else:
            print(f"[WARN] Known word '{kw}' not in Words sheet — skipped.")

    print(f"[INFO] {len(words)} words | B={budget} | known={len(known_indices)} | MIPGap={mipgap} | TimeLimit={tlimit}s")

    embeddings = load_embeddings(EMBED_FILE, words)
    dist       = build_distance_matrix(words, embeddings)
    print(f"[INFO] Distance matrix built ({dist.shape[0]}x{dist.shape[1]}, L2).")

    print(f"[INFO] Solving p-Median (B={budget}) ...")
    model, x, y, n = build_and_solve(words, weights, dist, budget, mipgap, tlimit, known_indices)
    write_results(model, words, weights, dist, x, y, n, budget, input_file)
