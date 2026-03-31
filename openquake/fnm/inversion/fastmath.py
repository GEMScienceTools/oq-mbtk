# ------------------- The OpenQuake Model Building Toolkit --------------------
# ------------------- FERMI: Fault nEtwoRks ModellIng -------------------------
# Copyright (C) 2023 GEM Foundation
#         .-.
#        /    \                                        .-.
#        | .`. ;    .--.    ___ .-.     ___ .-. .-.   ( __)
#        | |(___)  /    \  (   )   \   (   )   '   \  (''")
#        | |_     |  .-. ;  | ' .-. ;   |  .-.  .-. ;  | |
#       (   __)   |  | | |  |  / (___)  | |  | |  | |  | |
#        | |      |  |/  |  | |         | |  | |  | |  | |
#        | |      |  ' _.'  | |         | |  | |  | |  | |
#        | |      |  .'.-.  | |         | |  | |  | |  | |
#        | |      '  `-' /  | |         | |  | |  | |  | |
#       (___)      `.__.'  (___)       (___)(___)(___)(___)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import math
import numpy as np
import numba as nb


@nb.njit
def cscmatvec(n_col, Ap, Ai, Ax, Xx, Yx):
    for j in range(n_col):
        col_start = Ap[j]
        col_end = Ap[j + 1]

        for ii in range(col_start, col_end):
            i = Ai[ii]
            Yx[i] += Ax[ii] * Xx[j]


@nb.njit(parallel=True)
def cscmatvec_p(n_col, Ap, Ai, Ax, Xx, Yx):
    for i in nb.prange(Yx.size):
        Yx[i] = 0.0
    for j in nb.prange(n_col):
        col_start = Ap[j]
        col_end = Ap[j + 1]

        for ii in range(col_start, col_end):
            i = Ai[ii]
            Yx[i] += Ax[ii] * Xx[j]


@nb.njit(fastmath=True, parallel=True)
def spspmm_csr(A_data, A_indices, A_indptr, x, out):
    # out = A @ x   (CSR • dense vector)
    m = A_indptr.size - 1
    for i in nb.prange(m):
        row_sum = 0.0
        for j in range(A_indptr[i], A_indptr[i + 1]):
            row_sum += A_data[j] * x[A_indices[j]]
        out[i] = row_sum


@nb.njit()
def spspmm_csc(A_data, A_indices, A_indptr, x, out):
    n = len(x)
    # for i in nb.prange(out.size):
    #    out[i] = 0.0
    cscmatvec_p(n, A_indptr, A_indices, A_data, x, out)


# ------------------------------------------
# Core: CSC • dense vector (thread-safe)
# Uses atomic adds to avoid races
# ------------------------------------------
@nb.njit(fastmath=True, parallel=True, cache=True)
def spspmm_csc_atomic(A_data, A_indices, A_indptr, x, out):
    # out = A @ x (CSC) using atomics for row accumulation
    # IMPORTANT: out must be zeroed by caller.
    n = x.size  # number of columns
    for j in nb.prange(n):
        xj = x[j]
        c0 = A_indptr[j]
        c1 = A_indptr[j + 1]
        for p in range(c0, c1):
            i = A_indices[p]
            nb.atomic.add(out, i, A_data[p] * xj)


@nb.njit(fastmath=True, parallel=True, cache=True)
def project_to_min(vec, min: np.float64 = 0.0):
    for i in nb.prange(vec.size):
        if vec[i] < min:
            vec[i] = min


@nb.njit(fastmath=True, parallel=True, cache=True)
def norm2(x):
    acc = 0.0
    for i in nb.prange(x.size):
        acc += x[i] * x[i]
    return math.sqrt(acc)


@nb.njit(fastmath=True, parallel=True, cache=True)
def residual_norm_csr(A_data, A_indices, A_indptr, y, b):
    m = b.size
    acc = 0.0
    # Reduction across rows; each thread accumulates local then atomically adds.
    # Simpler: use a private accumulator per thread and reduce after.
    # Numba doesn't expose thread id easily, so we just atomic-add once per row.
    for i in nb.prange(m):
        s = 0.0
        row_start = A_indptr[i]
        row_end = A_indptr[i + 1]
        for p in range(row_start, row_end):
            s += A_data[p] * y[A_indices[p]]
        r = s - b[i]
        # Accumulate square to global
        nb.atomic.add(np.array([acc]), 0, r * r)  # tiny atomic trick
    return math.sqrt(acc)


@nb.njit(fastmath=True, parallel=True, cache=True)
def residual_norm_csc(A_data, A_indices, A_indptr, y, b):
    # compute Ay into a scratch, then norm of (Ay - b)
    m = b.size
    Ay = np.zeros(m, dtype=np.float64)
    spspmm_csc_atomic(A_data, A_indices, A_indptr, y, Ay)
    acc = 0.0
    for i in nb.prange(m):
        r = Ay[i] - b[i]
        acc += r * r
    return math.sqrt(acc)


@nb.njit(fastmath=True, parallel=True, cache=True)
def projected_grad_ratio(y, g, b_size):
    acc = 0.0
    for i in nb.prange(y.size):
        m = y[i] if y[i] < g[i] else g[i]
        acc += m * m
    return math.sqrt(acc) / b_size


def spmat_vec_mul(lhs, vec):
    M, N = lhs.shape
    result = np.zeros(M)

    cscmatvec(N, lhs.indptr, lhs.indices, lhs.data, vec, result)

    return result


@nb.njit(parallel=True)
def _csc_multivec_mul(n_col, Ap, Ai, Ax, Xxs, Yxs):
    NV = Yxs.shape[0]

    for i in nb.prange(NV):
        vec = Xxs[i, :]
        res = Yxs[i, :]
        cscmatvec(n_col, Ap, Ai, Ax, vec, res)


def spmat_multivec_mul(lhs, vecs):
    M, N = lhs.shape
    NV = vecs.shape[0]
    result = np.zeros((NV, M))
    # print(result.shape)

    _csc_multivec_mul(N, lhs.indptr, lhs.indices, lhs.data, vecs, result)

    return result
