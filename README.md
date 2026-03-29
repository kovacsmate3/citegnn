# CiteGNN

Link prediction in citation networks using classical graph methods and graph neural networks.

Current scope:

- datasets: `Cora`, `PubMed`
- task: citation link prediction
- framework: PyTorch Geometric
- package manager: `uv`

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

Current CPU configuration in [pyproject.toml](/var/home/anagy/repos/citegnn/pyproject.toml):

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
- preserved `train_mask`, `val_mask`, `test_mask`
- link-prediction split artifact generation
- smoke-test CLI

Main entrypoint:

```bash
python scripts/build_link_prediction_split.py --dataset Cora
python scripts/build_link_prediction_split.py --dataset PubMed
```

## Notes

The split policy for link prediction is still an open protocol decision.
The current pipeline preserves the original Planetoid node masks and generates link-prediction splits separately.
Before experiments are finalized, the split strategy should be checked against the papers cited in [proposal/references.bib](/var/home/anagy/repos/citegnn/proposal/references.bib).
