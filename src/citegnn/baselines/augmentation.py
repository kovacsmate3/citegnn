"""Lightweight structural / positional features for the GNN ablation.

Implements the secondary research lever from the project proposal
(\"Do lightweight structural or positional augmentations improve simple
GNN baselines?\"). Two complementary signals are concatenated to the raw
node attributes before the encoder:

* **Log-degree** -- ``log(1 + deg(v))`` of every node in the training
  message-passing graph. This is the cheapest possible structural
  encoding and surfaces hub-vs-leaf information that GCN's symmetric
  normalization smooths away.
* **Top-k Laplacian eigenvectors** -- the smallest-eigenvalue
  eigenvectors of the symmetric normalized Laplacian
  ``L_sym = I - D^{-1/2} A D^{-1/2}``. These are widely used as
  positional encodings (see e.g. Dwivedi et al., 2020). Sign ambiguity
  is broken deterministically by forcing the largest-magnitude entry of
  every eigenvector to be positive.

The augmented features are produced from the *training adjacency only*
to keep the message-passing graph and the structural features in sync
and to avoid any leakage from val/test edges.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse as sp
from scipy.sparse import linalg as splinalg
from torch import Tensor


def _build_train_adjacency_symmetric(
    train_edge_index: Tensor,
    *,
    num_nodes: int,
) -> sp.csr_matrix:
    """Build a symmetric 0/1 CSR adjacency from a (already symmetric) edge index."""

    src = train_edge_index[0].cpu().numpy()
    dst = train_edge_index[1].cpu().numpy()
    data = np.ones(src.shape[0], dtype=np.float32)
    A = sp.coo_matrix((data, (src, dst)), shape=(num_nodes, num_nodes)).tocsr()
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def _log_degree_feature(A: sp.csr_matrix) -> np.ndarray:
    degrees = np.asarray(A.sum(axis=1)).flatten()
    return np.log1p(degrees).astype(np.float32).reshape(-1, 1)


def _normalized_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    degrees = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(degrees)
    nonzero = degrees > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    I = sp.eye(A.shape[0], format="csr")
    return I - D_inv_sqrt @ A @ D_inv_sqrt


def _laplacian_eigenvectors(
    A: sp.csr_matrix,
    *,
    num_eigvecs: int,
) -> np.ndarray:
    """Return the ``num_eigvecs`` smallest non-trivial Laplacian eigenvectors."""

    if num_eigvecs <= 0:
        return np.zeros((A.shape[0], 0), dtype=np.float32)

    L = _normalized_laplacian(A)
    # We want the smallest eigenvalues. The first eigenvalue of the
    # symmetric normalized Laplacian is always 0 (or 0 per connected
    # component) and the corresponding eigenvector is trivial; we ask
    # for ``num_eigvecs + 1`` and drop the very first.
    k = min(num_eigvecs + 1, A.shape[0] - 1)
    if k <= 1:
        return np.zeros((A.shape[0], 0), dtype=np.float32)
    try:
        eigvals, eigvecs = splinalg.eigsh(L, k=k, which="SM", maxiter=2000)
    except splinalg.ArpackNoConvergence as exc:
        eigvals = exc.eigenvalues
        eigvecs = exc.eigenvectors
        if eigvecs.shape[1] == 0:
            return np.zeros((A.shape[0], num_eigvecs), dtype=np.float32)
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]
    eigvecs = eigvecs[:, 1 : 1 + num_eigvecs]

    # Resolve eigenvector sign ambiguity deterministically.
    for col in range(eigvecs.shape[1]):
        idx = int(np.argmax(np.abs(eigvecs[:, col])))
        if eigvecs[idx, col] < 0:
            eigvecs[:, col] = -eigvecs[:, col]

    if eigvecs.shape[1] < num_eigvecs:
        pad = np.zeros((eigvecs.shape[0], num_eigvecs - eigvecs.shape[1]), dtype=np.float32)
        eigvecs = np.hstack([eigvecs.astype(np.float32), pad])
    return eigvecs.astype(np.float32)


def compute_structural_features(
    *,
    num_nodes: int,
    train_edge_index: Tensor,
    num_eigvecs: int = 10,
) -> Tensor:
    """Return ``[N, 1 + num_eigvecs]`` log-degree + Laplacian PE features."""

    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if num_eigvecs < 0:
        raise ValueError("num_eigvecs must be non-negative")

    A = _build_train_adjacency_symmetric(train_edge_index, num_nodes=num_nodes)
    log_degree = _log_degree_feature(A)
    laplacian_pe = _laplacian_eigenvectors(A, num_eigvecs=num_eigvecs)
    if laplacian_pe.shape[1] == 0:
        combined = log_degree
    else:
        combined = np.hstack([log_degree, laplacian_pe])
    return torch.from_numpy(combined.astype(np.float32))


def augment_features(
    node_features: Tensor,
    *,
    structural: Tensor,
) -> Tensor:
    """Concatenate raw node features with structural features along dim=1."""

    if node_features.dim() != 2:
        raise ValueError("node_features must be two-dimensional")
    if structural.dim() != 2:
        raise ValueError("structural must be two-dimensional")
    if node_features.size(0) != structural.size(0):
        raise ValueError("node_features and structural must agree on the node count")
    return torch.cat(
        [node_features, structural.to(device=node_features.device, dtype=node_features.dtype)],
        dim=1,
    )


__all__ = [
    "augment_features",
    "compute_structural_features",
]
