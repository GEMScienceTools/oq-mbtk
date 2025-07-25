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


def similarity_csr_from_arrays(
        indices:  np.ndarray,
        indptr:   np.ndarray,
        data:     np.ndarray,
        *,
        N:        int | None = None,
        symmetric: bool = True
    ) -> sp.csr_matrix:
    """
    Assemble a CSR similarity/adjacency matrix.

    Parameters
    ----------
    indices, indptr, data
        The three 1‑D arrays that define a CSR matrix
        (exactly what `build_similarity_topk` returns).
    N : int, optional
        Number of columns/rows.  If ``None`` it is inferred from
        ``len(indptr) - 1``.
    symmetric : bool, default True
        Ensure `S` is perfectly symmetric by replacing it with
        ``S.maximum(S.T)``.  Leave False if you *know* the input is
        already symmetric and want to save a pass.

    Returns
    -------
    S : scipy.sparse.csr_matrix, shape = (N, N)
        Symmetric K‑nearest‑neighbour graph with cosine weights.
        The dtype is inherited from `data`.
    """
    if N is None:
        N = len(indptr) - 1

    S = sp.csr_matrix((data, indices, indptr), shape=(N, N))

    if symmetric:
        S = S.maximum(S.T)          # keep the larger weight on each edge

    return S


# ----------------------------------------------------------------------
# Convenience wrapper that does both steps in one call
# ----------------------------------------------------------------------
def build_similarity_sparse(A: sp.spmatrix, K: int, *, symmetric=True):
    """
    Complete helper:  calls `build_similarity_topk` (numba kernel) and
    returns a CSR similarity matrix.

        S = build_similarity_sparse(A, K)

    Parameters
    ----------
    A : scipy.sparse matrix  (M × N)
        Original data‑constraint matrix (columns = events).
    K : int
        Number of nearest neighbours per column to keep.
    symmetric : bool, default True
        Force symmetry with `S.maximum(S.T)`.

    Returns
    -------
    S : scipy.sparse.csr_matrix, shape = (N, N)
        Cosine‑similarity graph (weights in [0, 1]).
    """
    A = A.tocsr()
    indices_S, indptr_S, data_S = build_similarity_topk(A, K)
    return similarity_csr_from_arrays(indices_S, indptr_S, data_S,
                                      N=A.shape[1], symmetric=symmetric)

# -----------------------------------------------------------------------------
# 4.  System augmentation factory
# -----------------------------------------------------------------------------


def augment_system(
    A: sp.csr_matrix,
    d: np.ndarray,
    pi: np.ndarray,
    *,
    K: int = 10,
    lambda_P: float = 1.0,
    lambda_S: float = 0.1,
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
    A_aug = sp.vstack([A, lambda_P * P, lambda_S * B], format="csr")
    d_aug = np.concatenate(
        [
            d.astype(np.float32),
            lambda_P * b_P,
            np.zeros(B.shape[0], dtype=np.float32),
        ]
    )
    return A_aug, d_aug


def plausibility_residual(r, pi, *, scale_mode="log"):
    if scale_mode == "log":
        # P enforced   log(r_i) = log(pi_i)
        return np.log(r) - np.log(pi)
    else:              # linear mode
        return r - pi


def similarity_residual(r, indices_S, indptr_S):
    # quickly walk the sparse upper‑triangle adjacency
    i, j = [], []
    for row in range(len(indptr_S) - 1):
        for ptr in range(indptr_S[row], indptr_S[row+1]):
            col = indices_S[ptr]
            if row < col:          # keep strict upper triangle
                i.append(row)
                j.append(col)
    i = np.asarray(i, dtype=np.int32)
    j = np.asarray(j, dtype=np.int32)
    return r[i] - r[j]             # shape (n_edges,)


def neighbor_ratio_spread(r, pi, indices_S, indptr_S, scale_mode="log", eps=1e-30):
    if scale_mode == "log":
        # protect both vectors
        safe_r  = np.maximum(r,  eps)
        safe_pi = np.maximum(pi, eps)
        q = np.log(safe_r) - np.log(safe_pi)
    else:   # linear
        safe_pi = np.maximum(pi, eps)
        q = r / safe_pi

    edges_qdiff = []
    for row in range(len(indptr_S)-1):
        for k in range(indptr_S[row], indptr_S[row+1]):
            col = indices_S[k]
            diff = q[row] - q[col]
            if np.isfinite(diff):
                edges_qdiff.append(diff)

    edges_qdiff = np.asarray(edges_qdiff, dtype=np.float32)

    if edges_qdiff.size == 0:
        return {
            "rms_q_diff": np.nan,
            "p95_abs_q_diff": np.nan,
            "histogram": (np.array([]), np.array([]))
        }

    return {
        "rms_q_diff": np.sqrt(np.mean(edges_qdiff**2)),
        "p95_abs_q_diff": np.percentile(np.abs(edges_qdiff), 95),
        "histogram": np.histogram(edges_qdiff, bins=50)
    }


def plausibility_similarity_report(r, pi, S, *,
                                   scale_mode="log", lambda_P=1.0, lambda_S=0.1):

    e_P = plausibility_residual(r, pi, scale_mode=scale_mode)
    e_S = similarity_residual(r, S.indices, S.indptr)
    q_spread = neighbor_ratio_spread(r, pi, S.indices, S.indptr,
                                     scale_mode=scale_mode)
    print(f"P‑block:  RMS={np.sqrt(np.mean(e_P**2)):.3e}, ",
          f"max|e_P|={np.max(np.abs(e_P)):.3e}")
    print(f"S‑block:  RMS={np.sqrt(np.mean(e_S**2)):.3e},  ",
          f"max|e_S|={np.max(np.abs(e_S)):.3e}")
    print(f"Neighbor ratio RMS spread = {q_spread['rms_q_diff']:.3e}")


def proportionality_block(S,                   # sparse (N×N)  adjacency
                          pi,                  # (N,) plausibility, >0
                          lambda_P=1.0,        # weight for the whole block
                          use_weights=False    # step‑2 option
                          ):
    """
    Build an N×N sparse matrix B such that

        (B @ r)[i]  =  -r_i + Σ_j (p_i/p_j)·r_j                (unweighted)
        (B @ r)[i]  =  -Σ_j w_ij·r_i + Σ_j w_ij(p_i/p_j)·r_j   (weighted)

    equals zero when every rupture in each neighbourhood is proportional
    to its plausibility.

    Parameters
    ----------
    S : scipy.sparse matrix (N×N)
        k‑nearest‑neighbour adjacency, usually the output of
        `build_similarity_sparse`.  Only the *pattern* (and optionally the
        data) are used.
    pi : (N,) array_like
        Plausibility values, strictly positive.
    lambda_P : float, optional
        Scalar weight.  The returned matrix is `lambda_P * B_raw`.
    use_weights : bool, optional
        *False* (default)  → ignore `S.data`, use plain ratios.  
        *True*            → multiply every neighbour term by `w_ij`
                            and the r_i term by `-Σ w_ij`.

    Returns
    -------
    B : scipy.sparse.csr_matrix (N×N)
        Constraint block with ~N·(k+1) non‑zeros.
    """
    pi = np.asarray(pi, dtype=np.float32)
    if np.any(pi <= 0):
        raise ValueError("plausibility values must be positive")

    S_csr = S.tocsr()
    indptr, indices, weights = S_csr.indptr, S_csr.indices, S_csr.data
    N = S_csr.shape[0]

    # -------- build COO triplets -------------------------------------
    rows, cols, data = [], [], []

    for i in range(N):
        start, stop = indptr[i], indptr[i+1]
        nbr_idx = indices[start:stop]
        nbr_w   = weights[start:stop] if use_weights else None

        # ---- coefficient on r_i  ------------------------------------
        if use_weights:
            coeff_self = -np.sum(nbr_w, dtype=np.float32)
        else:
            coeff_self = -1.0
        rows.append(i)
        cols.append(i)
        data.append(lambda_P * coeff_self)

        # ---- neighbour coefficients --------------------------------
        p_i = pi[i]
        for jj, j in enumerate(nbr_idx):
            ratio = p_i / pi[j]
            coeff = ratio * (nbr_w[jj] if use_weights else 1.0)
            rows.append(i)
            cols.append(j)
            data.append(lambda_P * coeff)

    B = sp.coo_matrix((data, (rows, cols)), shape=(N, N),
                      dtype=np.float32).tocsr()
    return B
