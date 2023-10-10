import numpy as np
from numba import njit, jit, prange
from scipy import sparse as ssp


@njit
def cscmatvec(n_col, Ap, Ai, Ax, Xx, Yx):
    for j in range(n_col):
        col_start = Ap[j]
        col_end = Ap[j + 1]

        for ii in range(col_start, col_end):
            i = Ai[ii]
            Yx[i] += Ax[ii] * Xx[j]


def spmat_vec_mul(lhs, vec):
    M, N = lhs.shape
    result = np.zeros(M)

    cscmatvec(N, lhs.indptr, lhs.indices, lhs.data, vec, result)

    return result


@njit(parallel=True)
def _csc_multivec_mul(n_col, Ap, Ai, Ax, Xxs, Yxs):
    NV = Yxs.shape[0]

    for i in prange(NV):
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
