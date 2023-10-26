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

import numpy as np
from numba import njit, prange


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
