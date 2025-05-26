import math
from typing import Tuple, Union, Literal

import numpy as np
import scipy.sparse as sp
from numba import njit, prange

# -----------------------------------------------------------------------------
# 1.  Internal helpers – normalisation & tiny K‑heap maintenance
# -----------------------------------------------------------------------------


def _normalise_columns(A_csr: sp.csr_matrix) -> sp.csr_matrix:
    """Return a *new* **CSR** matrix whose columns have unit 2‑norm.

    ``scipy.sparse`` converts to COO when broadcasting with a 1‑D array; we
    force the result back to CSR to guarantee ``.indptr``/``.indices`` attrs.
    Columns with zero norm are left as all‑zero (their norm is set to 1).
    """
    col_norms = np.sqrt(A_csr.power(2).sum(axis=0)).A1.astype(np.float32)
    col_norms[col_norms == 0.0] = 1.0  # avoid division by zero
    inv_norms = 1.0 / col_norms
    return A_csr.multiply(inv_norms).tocsr()  # «ensure CSR»


@njit(inline="always")
def _restore_min_root(vals: np.ndarray, idx: np.ndarray):
    """After the 0‑th element was overwritten, move the new minimum to index 0."""
    k = 0
    for i in range(1, vals.shape[0]):  # tiny loop, K ≤ 15
        if vals[i] < vals[k]:
            k = i
    if k != 0:
        vals[0], vals[k] = vals[k], vals[0]
        idx[0], idx[k] = idx[k], idx[0]


@njit(parallel=True, fastmath=True)
def _topk_cosine_kernel(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_cols: int,
    K: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Populate *fixed‑size* ``topk_val`` and ``topk_idx`` arrays.

    ``topk_val[i, 0]`` always stores the current **minimum** within the K best
    of column *i* (min‑heap trick, O(K) memory per column).
    """
    topk_val = np.full((n_cols, K), -1.0, dtype=np.float32)
    topk_idx = np.full((n_cols, K), -1, dtype=np.int32)

    n_rows = indptr.shape[0] - 1

    for r in prange(n_rows):
        start = indptr[r]
        stop = indptr[r + 1]
        row_len = stop - start

        for ii in range(row_len):
            ci = indices[start + ii]
            vi = data[start + ii]
            for jj in range(ii + 1, row_len):
                cj = indices[start + jj]
                vj = data[start + jj]
                s = vi * vj  # contribution to cosine dot‑product

                # update heap of column ci
                if s > topk_val[ci, 0]:
                    topk_val[ci, 0] = s
                    topk_idx[ci, 0] = cj
                    _restore_min_root(topk_val[ci], topk_idx[ci])

                # update heap of column cj
                if s > topk_val[cj, 0]:
                    topk_val[cj, 0] = s
                    topk_idx[cj, 0] = ci
                    _restore_min_root(topk_val[cj], topk_idx[cj])

    return topk_idx, topk_val


# -----------------------------------------------------------------------------
# 2.  Public – similarity KNN graph (cosine of columns of *A*)
# -----------------------------------------------------------------------------


def build_similarity_topk(
    A: sp.csr_matrix, K: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return **CSR arrays** of a symmetric cosine KNN adjacency.

    ``indices_S, indptr_S, data_S`` correspond to the *upper‑triangle* edges
    (we symmetrise later) – this saves half the memory.
    """
    if not sp.isspmatrix_csr(A):
        raise TypeError("A must be CSR")
    if K < 1:
        raise ValueError("K must be ≥1")

    A_norm = _normalise_columns(A)  # now guaranteed CSR
    indptr, indices, data = A_norm.indptr, A_norm.indices, A_norm.data
    N = A_norm.shape[1]

    topk_idx, topk_val = _topk_cosine_kernel(
        indptr, indices, data.astype(np.float32), N, int(K)
    )

    # Convert K‑best → COO edge lists
    ei, ej, ew = [], [], []
    for i in range(N):
        for k in range(K):
            w = topk_val[i, k]
            j = int(topk_idx[i, k])
            if w > 0.0 and j >= 0:
                ei.append(i)
                ej.append(j)
                ew.append(float(w))

    S_coo = sp.coo_matrix((ew, (ei, ej)), shape=(N, N))
    S_upper = sp.triu(S_coo, k=1)
    S_sym = S_upper + S_upper.T
    S_sym.sum_duplicates()

    return (
        S_sym.indices.astype(np.int32),
        S_sym.indptr.astype(np.int32),
        S_sym.data.astype(np.float32),
    )


# -----------------------------------------------------------------------------
# 3.  Pseudo‑observation blocks
# -----------------------------------------------------------------------------


def make_plausibility_block(
    pi: np.ndarray, *, scale_mode: Literal["log", "linear"] = "log"
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Return (P, b_P) where *P* is ``N×N`` diagonal CSR and *b_P* is RHS vector.
    ``pi`` must be 1‑D of length N with 0 < πᵢ ≤ 1.
    """
    if pi.ndim != 1:
        raise ValueError("pi must be 1‑D")
    if not np.all((pi > 0) & (pi <= 1)):
        raise ValueError("plausibility values must be in (0, 1]")

    N = pi.shape[0]
    P = sp.eye(N, format="csr", dtype=np.float32)

    if scale_mode == "log":
        b_P = np.log(pi).astype(np.float32)
    elif scale_mode == "linear":
        b_P = pi.astype(np.float32)
    else:
        raise ValueError("scale_mode must be 'log' or 'linear'")

    return P, b_P


def make_similarity_block(
    indices_S: np.ndarray, indptr_S: np.ndarray, *, N: int
) -> sp.coo_matrix:
    """Convert the CSR adjacency (only upper‑triangle actually stored) into a *row*‑wise
    block B where every edge ⇒ one row ``[ … +1 at i … –1 at j … ]``.
    """
    S_csr = sp.csr_matrix(
        (np.ones_like(indices_S, dtype=np.float32), indices_S, indptr_S),
        shape=(N, N),
    )
    # Extract *strict* upper‑triangle edges to avoid duplicates
    edges_i, edges_j = sp.triu(S_csr, k=1).nonzero()
    n_edges = edges_i.size

    data = np.empty(n_edges * 2, dtype=np.float32)
    rows = np.empty(n_edges * 2, dtype=np.int32)
    cols = np.empty(n_edges * 2, dtype=np.int32)

    for e in range(n_edges):
        rows[2 * e] = e  # +1 coeff
        cols[2 * e] = edges_i[e]
        data[2 * e] = 1.0

        rows[2 * e + 1] = e  # –1 coeff
        cols[2 * e + 1] = edges_j[e]
        data[2 * e + 1] = -1.0

    B = sp.coo_matrix(
        (data, (rows, cols)), shape=(n_edges, N), dtype=np.float32
    )
    return B


# -----------------------------------------------------------------------------
# 4.  System augmentation factory
# -----------------------------------------------------------------------------


def augment_system(
    A: sp.csr_matrix,
    d: np.ndarray,
    pi: np.ndarray,
    *,
    K: int = 10,
    λ_P: float = 1.0,
    λ_S: float = 0.1,
    scale_mode: Literal["log", "linear"] = "log"
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Stack *P* and *B* blocks underneath the original system ``A x = d``.

    Returns ``A_aug`` (CSR) and ``d_aug`` (1‑D float32).
    """
    if not sp.isspmatrix_csr(A):
        raise TypeError("A must be CSR")
    if d.ndim != 1 or d.shape[0] != A.shape[0]:
        raise ValueError("d must be 1‑D with length M (= rows of A)")

    # ------------------------------------------------------------------
    # 1. Similarity graph (indices/indptr/data)
    # ------------------------------------------------------------------
    indices_S, indptr_S, data_S = build_similarity_topk(A, K)
    N = A.shape[1]
    B = make_similarity_block(indices_S, indptr_S, N=N)

    # ------------------------------------------------------------------
    # 2. Plausibility block
    # ------------------------------------------------------------------
    P, b_P = make_plausibility_block(pi, scale_mode=scale_mode)

    # ------------------------------------------------------------------
    # 3. Stack rows
    # ------------------------------------------------------------------
    A_aug = sp.vstack([A, λ_P * P, λ_S * B], format="csr")
    d_aug = np.concatenate(
        [
            d.astype(np.float32),
            λ_P * b_P,
            np.zeros(B.shape[0], dtype=np.float32),
        ]
    )
    return A_aug, d_aug
