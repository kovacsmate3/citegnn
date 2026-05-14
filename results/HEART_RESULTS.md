# HeaRT-Aligned Link Prediction Results

Each row aggregates the JSON outputs in `results/<dataset>/`. Values are means (± sample standard deviation) across the available seeds; rerun with additional `--seed` values to tighten the std estimate. The reference 'Paper MRR' / 'Paper Hits@10' come from **Table 4 of Li et al. (NeurIPS 2023)** "Evaluating Graph Neural Networks for Link Prediction".

### Cora

All metrics are reported in percent (mean ± std across seeds).
The 'Paper' column is the value reported by Li et al. (NeurIPS 2023) Table 4.
'Δ MRR' is ours minus paper.

| Method | MRR | Hits@1 | Hits@3 | Hits@10 | Hits@100 | Paper MRR | Paper Hits@10 | Δ MRR |
|---|---|---|---|---|---|---|---|---|
| Common Neighbors |  9.46 |  3.98 | 10.04 | 19.70 | 43.18 |  9.78 | 20.11 | -0.32 |
| Adamic-Adar | 11.43 |  4.73 | 12.50 | 25.57 | 42.23 | 11.91 | 24.10 | -0.48 |
| Personalized PageRank | 20.01 | 10.61 | 22.92 | 39.58 | 66.29 | 13.34 | 28.65 | +6.67 |
| Node2Vec |  3.38 |  0.76 |  2.27 |  7.01 | 38.07 | 14.47 | 32.77 | -11.09 |
| Feature-Only MLP |  1.33 |  0.00 |  0.19 |  0.95 | 30.68 | -- | -- | -- |
| GCN |  4.53 |  0.38 |  2.65 | 10.23 | 64.02 | 16.61 | 36.26 | -12.08 |
| GraphSAGE |  5.51 |  0.95 |  3.79 | 13.83 | 63.83 | 14.74 | 34.65 | -9.23 |
| GAT | 12.49 |  4.92 | 11.93 | 29.17 | 71.78 | -- | -- | -- |
| GCN + structural PE |  1.78 |  0.38 |  0.57 |  1.52 | 34.66 | -- | -- | -- |
| GraphSAGE + structural PE |  5.12 |  1.33 |  3.41 | 12.31 | 52.65 | -- | -- | -- |
| GAT + structural PE | 10.92 |  3.79 | 10.61 | 24.05 | 70.08 | -- | -- | -- |

### PubMed

All metrics are reported in percent (mean ± std across seeds).
The 'Paper' column is the value reported by Li et al. (NeurIPS 2023) Table 4.
'Δ MRR' is ours minus paper.

| Method | MRR | Hits@1 | Hits@3 | Hits@10 | Hits@100 | Paper MRR | Paper Hits@10 | Δ MRR |
|---|---|---|---|---|---|---|---|---|
| Common Neighbors |  2.13 |  0.56 |  1.49 |  3.84 | 19.07 |  2.28 |  4.78 | -0.15 |
| Adamic-Adar |  2.37 |  0.74 |  1.69 |  4.33 | 19.09 |  2.63 |  5.71 | -0.26 |
| Personalized PageRank |  4.50 |  1.31 |  3.50 |  9.61 | 41.20 |  4.26 |  9.81 | +0.24 |
| Node2Vec | -- | -- | -- | -- | -- |  3.94 |  8.51 | -- |
| Feature-Only MLP |  0.54 |  0.00 |  0.02 |  0.16 |  7.22 | -- | -- | -- |
| GCN |  3.38 |  0.56 |  1.67 |  6.52 | 54.26 |  7.13 | 15.22 | -3.75 |
| GraphSAGE |  3.07 |  0.23 |  1.33 |  5.73 | 57.63 |  9.39 | 20.54 | -6.32 |
| GAT |  2.29 |  0.23 |  0.79 |  3.47 | 47.83 | -- | -- | -- |
| GCN + structural PE |  2.57 |  0.25 |  1.20 |  5.10 | 41.02 | -- | -- | -- |
| GraphSAGE + structural PE |  1.75 |  0.05 |  0.43 |  2.66 | 37.18 | -- | -- | -- |
| GAT + structural PE |  1.93 |  0.07 |  0.74 |  3.41 | 37.48 | -- | -- | -- |
