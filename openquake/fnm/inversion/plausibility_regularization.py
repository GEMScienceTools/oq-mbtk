"""Experimental regularizers for linear inversions of rupture occurrence rates."""

from typing import Tuple, Literal

import numpy as np
import scipy.sparse as sp


def _normalise_columns(A_csr):
    A_csr = sp.csr_matrix(A_csr, dtype=float)
    norms = np.sqrt(np.asarray(A_csr.power(2).sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    return A_csr.multiply(1.0 / norms).tocsr()


def build_similarity_topk(A, K=10, **kwargs):
    """Return indices, indptr, data of the symmetric cosine neighbor graph.

    See ``build_similarity_sparse`` for candidate restrictions. Symmetrization
    keeps the union of directed selections, so final degrees can exceed K.
    """
    S = build_similarity_sparse(A, K, **kwargs)
    return S.indices, S.indptr, S.data


# -----------------------------------------------------------------------------
# 3.  Pseudo‑observation blocks
# -----------------------------------------------------------------------------


def make_plausibility_block(
    pi: np.ndarray, *, scale_mode: Literal["log", "linear"] = "log",
    rate_scale: float | None = None
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Explicit absolute-rate prior; use proportionality_block for relative rates.

    Linear mode targets ``r = rate_scale * pi``. Log mode is rejected because
    stacking an identity matrix cannot impose a logarithm of the unknown rates.
    """
    pi = np.asarray(pi, dtype=float)
    if pi.ndim != 1 or not np.all(np.isfinite(pi) & (pi > 0) & (pi <= 1)):
        raise ValueError("pi must be a finite 1-D array in (0, 1]")
    if scale_mode != "linear":
        raise ValueError("log-rate priors are not linear rate equations; use proportionality_block")
    if rate_scale is None or not np.isfinite(rate_scale) or rate_scale <= 0:
        raise ValueError("linear absolute-rate priors require a positive rate_scale")
    return sp.eye(len(pi), format="csr"), rate_scale * pi


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
def build_similarity_sparse(
    A, K=10, *, symmetric=True, min_similarity=0.0,
    magnitudes=None, max_mag_difference=None, batch_size=256,
):
    """Exact positive column cosines, restricted before selecting K neighbors.

    Accepts sparse matrices and arrays. Products are evaluated in batches to
    avoid allocating the complete N-by-N Gram matrix. Dense input rows can
    still make this expensive; use fault participation rows, not regional MFD
    rows, when similarity is intended to represent shared fault geometry.

    Optional magnitude differences and ``min_similarity`` restrict candidates
    *before* top-K selection. Ties are broken by column index. The symmetric
    graph is the union of selections, preserving either direction of an edge.
    """
    if not sp.issparse(A):
        raise TypeError("A must be a sparse matrix or array")
    if not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError("K must be a positive integer")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(min_similarity) or not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be in [0, 1]")
    A = sp.csr_matrix(A, dtype=float, copy=True)
    A.sum_duplicates()
    if not np.all(np.isfinite(A.data)):
        raise ValueError("A must contain finite values")
    N = A.shape[1]
    if max_mag_difference is not None:
        magnitudes = np.asarray(magnitudes, dtype=float)
        if magnitudes.shape != (N,) or not np.all(np.isfinite(magnitudes)):
            raise ValueError("magnitudes must be finite and aligned with A columns")
        if not np.isfinite(max_mag_difference) or max_mag_difference < 0:
            raise ValueError("max_mag_difference must be finite and nonnegative")
    A_norm = _normalise_columns(A)
    columns = A_norm.tocsc()
    rows, cols, values = [], [], []
    for start in range(0, N, batch_size):
        products = (columns[:, start:start + batch_size].T @ A_norm).tocsr()
        for local in range(products.shape[0]):
            i = start + local
            lo, hi = products.indptr[local:local + 2]
            js = products.indices[lo:hi]
            scores = np.clip(products.data[lo:hi], -1.0, 1.0)
            keep = (js != i) & (scores > 0) & (scores >= min_similarity)
            if max_mag_difference is not None:
                keep &= np.abs(magnitudes[js] - magnitudes[i]) <= max_mag_difference + 1e-12
            js, scores = js[keep], scores[keep]
            best = np.lexsort((js, -scores))[:K]
            rows.extend([i] * len(best))
            cols.extend(js[best])
            values.extend(scores[best])
    S = sp.csr_matrix((values, (rows, cols)), shape=(N, N))
    if symmetric:
        S = S.maximum(S.T).tocsr()
    S.sort_indices()
    return S

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
    scale_mode: Literal["log", "linear"] = "log",
    rate_scale: float | None = None,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Stack *P* and *B* blocks underneath the original system ``A x = d``.

    This legacy absolute-prior helper requires explicit linear mode and
    rate_scale. For relative plausibility, stack proportionality_block instead.
    Returns ``A_aug`` (CSR) and ``d_aug`` (1-D float64).
    """
    A = sp.csr_matrix(A, dtype=float)
    d = np.asarray(d, dtype=float)
    P, b_P = make_plausibility_block(pi, scale_mode=scale_mode, rate_scale=rate_scale)
    if P.shape[1] != A.shape[1]:
        raise ValueError("pi must be aligned with A columns")
    if d.ndim != 1 or d.shape[0] != A.shape[0]:
        raise ValueError("d must be 1‑D with length M (= rows of A)")

    # ------------------------------------------------------------------
    # 1. Similarity graph (indices/indptr/data)
    # ------------------------------------------------------------------
    indices_S, indptr_S, data_S = build_similarity_topk(A, K)
    N = A.shape[1]
    B = make_similarity_block(indices_S, indptr_S, N=N)

    # ------------------------------------------------------------------
    # 3. Stack rows
    # ------------------------------------------------------------------
    A_aug = sp.vstack([A, lambda_P * P, lambda_S * B], format="csr")
    d_aug = np.concatenate(
        [
            d.astype(float),
            lambda_P * b_P,
            np.zeros(B.shape[0], dtype=np.float32),
        ]
    )
    return A_aug, d_aug


def plausibility_residual(r, pi, *, scale_mode="log"):
    if scale_mode == "log":
        # Descriptive log-ratio diagnostic, NOT a linear-system residual.
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


def plausibility_similarity_report(
    r, pi, S, *, scale_mode="log", lambda_P=1.0, lambda_S=0.1,
    block=None, formulation="laplacian", use_weights=False,
):
    """Report actual proportionality residuals and descriptive neighbor spread.

    Pass the exact ``block`` used in the inversion (including its row scaling)
    or specify its construction options. ``scale_mode`` affects only the
    descriptive ratio spread, not the linear-system residual. lambda_S applies
    to the separate ordinary-smoothing diagnostic, if that block is used.
    """
    r = np.asarray(r, dtype=float)
    if block is None:
        block = proportionality_block(S, pi, lambda_P, use_weights,
                                      formulation=formulation)
    e_P = np.asarray(block @ r).ravel()
    e_S = lambda_S * similarity_residual(r, S.indices, S.indptr)
    for label, e in (("Proportionality block", e_P), ("Ordinary smoothing", e_S)):
        rms = np.sqrt(np.mean(e**2)) if e.size else 0.0
        largest = np.max(np.abs(e)) if e.size else 0.0
        print(f"{label}: RMS={rms:.3e}, max|residual|={largest:.3e}")
    q_spread = neighbor_ratio_spread(r, pi, S.indices, S.indptr,
                                     scale_mode=scale_mode)
    print(f"Neighbor ratio RMS spread = {q_spread['rms_q_diff']:.3e}")


def proportionality_block(
    S, pi, lambda_P=1.0, use_weights=False, *, formulation="laplacian",
):
    """Linear constraints with zero RHS favoring rates proportional to pi.

    ``laplacian`` preserves the original N-by-N interface, with corrected
    diagonal: -sum(w_ij)*r_i + sum(w_ij*pi_i/pi_j*r_j). An isolated rupture
    has a zero row. Unweighted mode uses w_ij=1 for each existing edge.

    ``pairwise`` uses one row per undirected edge, with coefficients
    sqrt(w_ij) * [pi_j, -pi_i] / hypot(pi_i, pi_j). This avoids cancellation
    between neighbors and divisions by tiny plausibilities. Its row norm is
    sqrt(w_ij), independent of the common plausibility scale. For a symmetric S, it has the same
    nullspace as the Laplacian, but a different penalty away from it.

    Both modes annihilate r=c*pi. lambda_P multiplies rows, so its square
    multiplies the least-squares penalty. pi must be finite and positive;
    zero-plausibility ruptures need an explicit exclusion/fixed-zero policy.
    """
    S = sp.csr_matrix(S, dtype=float, copy=True)
    S.sum_duplicates()
    if S.shape[0] != S.shape[1]:
        raise ValueError("S must be square")
    pi = np.asarray(pi, dtype=float)
    if pi.shape != (S.shape[0],) or not np.all(np.isfinite(pi) & (pi > 0)):
        raise ValueError("pi must be finite, positive and aligned with S")
    if not np.all(np.isfinite(S.data) & (S.data >= 0)):
        raise ValueError("similarity weights must be finite and nonnegative")
    if not np.isfinite(lambda_P) or lambda_P < 0:
        raise ValueError("lambda_P must be finite and nonnegative")
    S.setdiag(0)
    S.eliminate_zeros()
    if not use_weights:
        S.data[:] = 1.0
    if formulation == "laplacian":
        degrees = np.asarray(S.sum(axis=1)).ravel()
        return (lambda_P * (sp.diags(pi) @ S @ sp.diags(1 / pi)
                            - sp.diags(degrees))).tocsr()
    if formulation != "pairwise":
        raise ValueError("formulation must be 'laplacian' or 'pairwise'")
    edges = sp.triu(S.maximum(S.T), k=1).tocoo()
    i, j = edges.row, edges.col
    # Rescale first so the normalization is stable even for very small pi.
    scale = np.maximum(pi[i], pi[j])
    p_i, p_j = pi[i] / scale, pi[j] / scale
    weights = lambda_P * np.sqrt(edges.data) / np.hypot(p_i, p_j)
    n = len(i)
    return sp.csr_matrix(
        (np.concatenate((weights * p_j, -weights * p_i)),
         (np.concatenate((np.arange(n), np.arange(n))), np.concatenate((i, j)))),
        shape=(n, len(pi)),
    )
