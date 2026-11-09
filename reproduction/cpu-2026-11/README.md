# CPU reproduction, November 2026

A rerun of the seed-1 HeaRT sweep, done after the course for the public mirror. It checks
the committed numbers in `results/cora/` and `results/pubmed/`, which are unchanged.
The rerun JSONs are in `cora/` and `pubmed/`, in the same format.

## Setup

- Code: `scripts/run_all_heart.py` with the recorded configurations (`RECORDED_CONFIGS`,
  copied from the `params` of the committed JSONs), seed 1, K = 500.
- Environment: `uv sync --locked` (Python 3.14, torch 2.11.0+cpu, torch-geometric 2.7.0).
- Machine: laptop, Intel Core i7-12700H, Windows 11, no GPU. The Cora and PubMed sweeps
  ran at the same time with `OMP_NUM_THREADS=8` each, so the timings are indicative only.

```bash
uv run python scripts/run_all_heart.py --dataset Cora --seed 1 --results-dir reproduction/cpu-2026-11
uv run python scripts/run_all_heart.py --dataset PubMed --seed 1 --skip node2vec --results-dir reproduction/cpu-2026-11
```

## Results

Test set, percent. "Committed" is `results/<dataset>/<method>_seed1.json`.

**Cora**

| Method | Committed MRR | Rerun MRR | Δ MRR | Committed Hits@10 | Rerun Hits@10 | Δ Hits@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Common Neighbors | 9.46 | 9.46 | ±0.00 | 19.70 | 19.70 | ±0.00 |
| Adamic-Adar | 11.43 | 11.43 | ±0.00 | 25.57 | 25.57 | ±0.00 |
| Personalized PageRank | 20.01 | 20.01 | ±0.00 | 39.58 | 39.58 | ±0.00 |
| Node2Vec | 3.38 | 3.47 | +0.09 | 7.01 | 6.44 | −0.57 |
| Feature-only MLP | 1.33 | 0.84 | −0.48 | 0.95 | 0.38 | −0.57 |
| GCN | 4.53 | 3.43 | −1.10 | 10.23 | 7.39 | −2.84 |
| GraphSAGE | 5.51 | 6.45 | +0.94 | 13.83 | 15.15 | +1.33 |
| GAT | 12.49 | 13.63 | +1.14 | 29.17 | 29.17 | ±0.00 |
| GCN + structural PE | 1.78 | 2.19 | +0.42 | 1.52 | 3.22 | +1.70 |
| GraphSAGE + structural PE | 5.12 | 3.86 | −1.26 | 12.31 | 10.23 | −2.08 |
| GAT + structural PE | 10.92 | 11.22 | +0.30 | 24.05 | 26.14 | +2.08 |

**PubMed**

| Method | Committed MRR | Rerun MRR | Δ MRR | Committed Hits@10 | Rerun Hits@10 | Δ Hits@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Common Neighbors | 2.13 | 2.13 | ±0.00 | 3.84 | 3.84 | ±0.00 |
| Adamic-Adar | 2.37 | 2.37 | ±0.00 | 4.33 | 4.33 | ±0.00 |
| Personalized PageRank | 4.50 | 4.50 | ±0.00 | 9.61 | 9.61 | ±0.00 |
| Feature-only MLP | 0.54 | 0.54 | ±0.00 | 0.16 | 0.16 | ±0.00 |
| GCN | 3.38 | 3.36 | −0.02 | 6.52 | 6.63 | +0.11 |
| GraphSAGE | 3.07 | 3.19 | +0.12 | 5.73 | 5.78 | +0.05 |
| GAT | 2.29 | 2.34 | +0.05 | 3.47 | 3.47 | ±0.00 |
| GCN + structural PE | 2.57 | 2.70 | +0.13 | 5.10 | 5.53 | +0.43 |
| GraphSAGE + structural PE | 1.75 | 1.78 | +0.04 | 2.66 | 2.44 | −0.23 |
| GAT + structural PE | 1.93 | 1.95 | +0.02 | 3.41 | 3.61 | +0.20 |

Node2Vec has no committed PubMed result and was skipped.

## What reproduces

- **Heuristics, exactly.** On Cora the whole `test` and `validation` metric dicts of CN,
  AA and PPR are identical to the committed ones. On PubMed they are too, except PPR's
  average precision, which differs in the 16th significant digit.
- **Feature-only MLP on PubMed**, to within 0.03 percentage points on every metric.
- **Trainable models, approximately.** They use exactly the recorded configuration
  (every `params` field matches except `best_epoch`), but training isn't bit-deterministic
  across runs and machines, so early stopping can pick a different epoch. On PubMed every
  GNN lands within 0.13 MRR points of the committed value, on Cora within 1.3.
- **The conclusions hold.** On PubMed the whole MRR ranking is unchanged, from PPR at the
  top to the feature-only MLP at the bottom. On Cora the top of the ranking is the same
  (PPR > GAT > Adamic-Adar > GAT + PE > Common Neighbors) and the MLP is last. On both
  datasets structural PE lowers MRR for all three encoders (Cora: GCN 3.43 → 2.19,
  GraphSAGE 6.45 → 3.86, GAT 13.63 → 11.22; PubMed: GCN 3.36 → 2.70, GraphSAGE
  3.19 → 1.78, GAT 2.34 → 1.95).

## Timings

| Stage | Cora | PubMed |
| --- | ---: | ---: |
| HeaRT split (incl. Planetoid download) | cached (48 s on first build) | ≈ 3 min |
| Common Neighbors / Adamic-Adar / PPR | 0.2 s / 0.1 s / 3.3 s | 1.7 s / 2.1 s / 74.5 s |
| Node2Vec | 180 s | skipped |
| Feature-only MLP | 34 s | 39 s |
| GCN / GraphSAGE / GAT | 226 s / 469 s / 327 s | 2,102 s / 407 s / 646 s |
| GCN / GraphSAGE / GAT + structural PE | 161 s / 299 s / 271 s | 852 s / 451 s / 606 s |
| **Full sweep** | **33 min** (1,972 s) | **89 min** (5,353 s, incl. split) |

The first 33 minutes of the PubMed sweep shared the CPU with the Cora sweep.
