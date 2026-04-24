================================================================================
  SET COVERING MATHEMATICAL OPTIMIZATION
  EMU414 - Mathematical Programming and Computer Applications
  Hacettepe University, Spring 2025-2026
================================================================================

--------------------------------------------------------------------------------
PYTHON VERSION
--------------------------------------------------------------------------------
Python 3.9 or higher is required.

--------------------------------------------------------------------------------
GUROBI VERSION
--------------------------------------------------------------------------------
Gurobi 10.x or higher is required.
An academic license can be obtained from: https://www.gurobi.com/academia/

--------------------------------------------------------------------------------
HOW TO RUN THE MODEL
--------------------------------------------------------------------------------
Step 1: Install dependencies
    pip install gurobipy numpy pandas gensim

Step 2: Prepare data (generates words_100.csv and distance_matrix.csv)
    python data/prepare_data.py

Step 3: Run the optimization model
    python model.py

Output files will be created automatically:
    - model.lp           (exported LP formulation)
    - gurobi.log         (solver log)
    - results/solution.txt  (optimal solution)

To change the budget or coverage radius, edit the constants at the top of
model.py:
    BUDGET = 10            # number of representative words to select
    COVERAGE_RADIUS = 0.3  # cosine distance threshold for coverage

--------------------------------------------------------------------------------
FILE DESCRIPTIONS
--------------------------------------------------------------------------------
README.md
    Full project description, mathematical formulation, and instructions.
    Intended for GitHub.

readme.txt
    This file. Submission instructions for course requirements.

model.py
    Main Gurobi ILP implementation. Loads data, builds and solves the
    Maximum Coverage model, exports model.lp, gurobi.log, and the solution.

model.lp
    LP file exported by Gurobi from model.py. Contains the full mathematical
    formulation in LP format. Generated automatically when model.py is run.

gurobi.log
    Raw Gurobi solver log. Contains branch-and-bound statistics, bound
    progress, and final optimality status. Generated automatically.

results/solution.txt
    Human-readable solution output: selected words, covered words, objective
    value, and coverage percentage. Generated automatically.

data/prepare_data.py
    Data preparation pipeline. Downloads/loads the top-10k English word list,
    selects 100 words, loads GloVe/Word2Vec embeddings, computes pairwise
    cosine distances, and saves words_100.csv and distance_matrix.csv.

data/words_100.csv
    The 100-word subset selected from the top-10k most frequent English words.
    Columns: rank, word. Input to model.py.

data/distance_matrix.csv
    100x100 symmetric matrix of pairwise cosine distances computed from
    GloVe/Word2Vec embeddings. Rows and columns correspond to words in
    words_100.csv. Input to model.py.

report/
    Directory containing all LaTeX source files for the mini paper
    (minipaper.tex and supporting files). Compile with pdflatex or Overleaf.

slides/
    Directory containing the presentation slides (slides.pdf).

--------------------------------------------------------------------------------
NOTES FOR REPRODUCIBILITY
--------------------------------------------------------------------------------
- The GloVe embeddings used are the pre-trained 100-dimensional vectors
  (glove.6B.100d). Download from: https://nlp.stanford.edu/projects/glove/
  Place the .txt file in the data/ directory before running prepare_data.py.

- All random seeds are fixed in prepare_data.py to ensure reproducibility.

- The model has been tested on Windows 11 with Python 3.9 and Gurobi 10.0.

================================================================================
