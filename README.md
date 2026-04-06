# CiteGNN

Link prediction in citation networks using classical graph methods and graph neural networks.

Current scope:

- datasets: `Cora`, `PubMed`
- task: citation link prediction
- framework: PyTorch Geometric
- package manager: `uv`
- status: data pipeline and shared evaluation utilities implemented

## Environment

The shared project configuration defaults to CPU PyTorch wheels through `uv`.

Key points:

- Python target: `>=3.14`
- package manager: `uv`
- default PyTorch source: `https://download.pytorch.org/whl/cpu`

Install the default environment:

```bash
uv lock
uv sync
```

Run the current checks:

```bash
uv run python -m compileall src scripts
uv run mypy src scripts
```

## CUDA Override

The repository is intentionally configured for CPU by default so CI and local development stay predictable.

If you have a CUDA-capable machine and want to use CUDA wheels locally, override the PyTorch source in your local checkout before syncing.
The project already declares both the CPU and CUDA indexes, so you only need to change `tool.uv.sources`.

Current CPU configuration in [pyproject.toml](pyproject.toml):

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]
```

Example local CUDA override for CUDA 12.8:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
torchvision = [{ index = "pytorch-cu128" }]
```

After changing `tool.uv.sources`, regenerate the lockfile and sync:

```bash
uv lock
uv sync
```

Then verify the installed backend:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Notes:

- This override is intended for local development on GPU machines.
- The shared repository and CI should stay on CPU unless the team explicitly decides to support CUDA in the common environment.
- If you switch between CPU and CUDA frequently, expect to re-lock and re-sync.

## Data Pipeline

Current implementation:

- dataset loading through `Planetoid`
- direct module-level imports for `torch` and `torch_geometric`
- preserved `train_mask`, `val_mask`, `test_mask`
- link-prediction split artifact generation
- default undirected link-prediction protocol with split leakage checks
- smoke-test CLI
- runtime verified on `Cora` and `PubMed`

Main entrypoint:

```bash
python scripts/build_link_prediction_split.py --dataset Cora
python scripts/build_link_prediction_split.py --dataset PubMed
```

Key modules:

- [src/citegnn/data/loaders.py](src/citegnn/data/loaders.py)
- [src/citegnn/data/splits.py](src/citegnn/data/splits.py)
- [src/citegnn/data/smoke.py](src/citegnn/data/smoke.py)
- [docs/link_prediction_protocol.md](docs/link_prediction_protocol.md)

Demonstration notebooks:

- [notebooks/data_pipeline_demo.ipynb](notebooks/data_pipeline_demo.ipynb)
- [notebooks/heuristic_baselines_demo.ipynb](notebooks/heuristic_baselines_demo.ipynb)

## Evaluation

Current implementation:

- shared binary link-prediction metrics
- validation-threshold selection
- reusable experiment result schema
- JSON result writing for later aggregation

Main modules:

- [src/citegnn/utils/metrics.py](src/citegnn/utils/metrics.py)
- [src/citegnn/utils/io.py](src/citegnn/utils/io.py)
- [src/citegnn/evaluation.py](src/citegnn/evaluation.py)

## Heuristic Baselines

Current implementation:

- heuristic scorers built from the training graph only
- supported methods:
  - `common_neighbors`
  - `adamic_adar`
  - `personalized_pagerank`
- shared evaluator and JSON result artifacts reused from the main evaluation layer

Run one baseline from a saved split:

```bash
python scripts/run_heuristic_baselines.py \
  --split-path artifacts/splits/cora_seed42_lp.pt \
  --method common_neighbors
```

Key modules:

- [src/citegnn/baselines/heuristics.py](src/citegnn/baselines/heuristics.py)
- [scripts/run_heuristic_baselines.py](scripts/run_heuristic_baselines.py)

## Node2Vec Baseline

Current implementation:

- train-only node2vec random walks
- lightweight PyTorch skip-gram trainer with negative sampling
- Hadamard edge features from learned node embeddings
- logistic-regression edge classifier with the shared evaluation layer

Run the baseline from a saved split:

```bash
python scripts/run_node2vec_baseline.py \
  --split-path artifacts/splits/cora_seed42_lp.pt
```

Key modules:

- [src/citegnn/baselines/node2vec.py](src/citegnn/baselines/node2vec.py)
- [scripts/run_node2vec_baseline.py](scripts/run_node2vec_baseline.py)

## Notes

The split policy for link prediction is still an open protocol decision.
The current pipeline preserves the original Planetoid node masks and generates link-prediction splits separately.
Before experiments are finalized, the split strategy should be checked against the papers cited in [proposal/references.bib](proposal/references.bib).
