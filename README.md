# Google Hash Code 2017 — Streaming Videos Optimization

Course project for Combinatorial and Network Optimization Algorithms exam, Master's Degree in Computer Engineering, University of Naples Federico II.

Solves the **Google Hash Code 2017 Qualification Round** problem ("Streaming Videos"): given a central data center holding every video, a set of cache servers with limited capacity, and endpoints (groups of users) each connected to the data center and to some subset of caches with different latencies, decide which videos to replicate on which caches to minimize average request latency — equivalently, to maximize the average time saved per request compared to always serving from the data center.

## Approach

Implemented in Python:

- **Greedy / marginal greedy** cache-filling heuristics (`run_greedy`, `run_greedy_marginal` in [`soluzione/algorithms2_copy.py`](soluzione/algorithms2_copy.py)) — videos are assigned to caches based on the expected latency saving per unit of cache space they occupy.
- **Local search** (`run_local_search`) — starting from a greedy solution, repeatedly tries swapping videos in/out of caches to improve the score, within configurable limits on the number of candidates considered per cache (for tractability on larger instances).
- [`soluzione/notebookSoluzione.ipynb`](soluzione/notebookSoluzione.ipynb) — a notebook walking through the problem definition, scoring function, data parsing, and the algorithm/results analysis (with the charts in [`soluzione/immagini/`](soluzione/immagini/)). Best place to start reading.
- [`soluzione/main.py`](soluzione/main.py) — standalone script entry point to run the solver on a given input file.

## Problem instances

The official qualification round provided 4 input files (in [`hashcode_2017_qualification_round.in/`](hashcode_2017_qualification_round.in/)): `kittens`, `me_at_the_zoo`, `trending_today`, `videos_worth_spreading` — ranging from a few endpoints/videos to much larger, more realistic scenarios. Results for each (under different parameter configurations: marginal greedy `M` vs. standard greedy with `K` local-search passes) are in [`output/`](output/).

Full problem statement: [`hashcode_2017_qualification_round.pdf`](hashcode_2017_qualification_round.pdf).
