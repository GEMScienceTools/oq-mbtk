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
from scipy import sparse as ssp
from scipy.sparse.linalg import svds

# from sklearn.linear_model import LinearRegression
from scipy.optimize import (
    nnls,
    dual_annealing,
    lsq_linear,
)


def weight_from_error(error, zero_error=1e-10):
    if error == 0:
        error = zero_error

    return 1 / error**2


def weights_from_errors(errors, zero_error=1e-10):
    return np.array([weight_from_error(error, zero_error) for error in errors])


def solve_dense_svd(A, d):
    # Compute the SVD of A
    U, sigma, Vt = np.linalg.svd(A, full_matrices=True)

    # Compute the pseudoinverse of A from the SVD
    # Create a diagonal matrix from sigma
    D_sigma = np.zeros_like(A, dtype=float)
    D_sigma[: A.shape[0], : A.shape[0]] = np.diag(sigma)

    # Compute the pseudoinverse of D_sigma
    D_sigma_pinv = np.zeros_like(A, dtype=float)
    non_zero_elements = D_sigma != 0
    D_sigma_pinv[non_zero_elements] = 1.0 / D_sigma[non_zero_elements]

    # Compute the pseudoinverse of A
    A_pinv = Vt.T @ D_sigma_pinv.T @ U.T

    # Compute a particular solution
    x0 = A_pinv @ d

    # Find the null space of A from the SVD
    null_space = Vt[sigma.size:]

    return x0, null_space


def solve_sparse_svd(A, d):
    # Compute the SVD of A using svds
    k = min(A.shape) - 1  # maximum number of singular values svds can compute
    U, sigma, Vt = svds(A, k=k)

    # Reverse the outputs, as svds returns them in ascending order
    U = U[:, ::-1]
    sigma = sigma[::-1]
    Vt = Vt[::-1, :]

    # Compute the pseudoinverse of A from the SVD
    D_sigma = np.diag(sigma)

    D_sigma_pinv = np.zeros_like(D_sigma)
    non_zero_elements = D_sigma != 0
    D_sigma_pinv[non_zero_elements] = 1.0 / D_sigma[non_zero_elements]

    A_pinv = Vt.T @ D_sigma_pinv @ U.T

    # Compute a particular solution
    x0 = A_pinv @ d

    # Find the null space of A from the SVD
    null_space = Vt[sigma.size:]

    return x0, null_space


def solve_svd(A, d, return_nullspace=False):
    print("solving w/ SVD")
    # if A_type == 'dense':
    if isinstance(A, np.ndarray):
        x0, null_space = solve_dense_svd(A, d)
    # elif A_type == 'sparse':
    elif ssp.issparse(A):
        x0, null_space = solve_sparse_svd(A, d)
    else:
        raise NotImplementedError("A must be dense or sparse")

    norm = np.linalg.norm(A @ x0 - d)
    print("norm", norm)

    if return_nullspace:
        return_vals = (x0, null_space)
    else:
        return_vals = x0

    return return_vals


def compute_gradient(G, GT, d, x, verbose=False):
    if verbose:
        print("G", G.shape)
        print("GT", GT.shape)
        print("x", x.shape)
        print("d", d.shape)

    pred = G.dot(x)
    if verbose:
        print("pred", pred.shape)

    residual = pred - d
    if verbose:
        print("residual", residual.shape)

    gradient = 2 * GT.dot(residual)

    return gradient


def gradient_descent_unweighted(
    G,
    d,
    x_init,
    alpha=0.01,
    alpha_decay=True,
    grad_perturb=False,
    num_iterations=10000,
    tol=1e-8,
    verbose=False,
    min_bounds=None,
    max_bounds=None,
):
    norms = np.zeros(num_iterations)

    x = x_init
    GT = G.transpose()

    if np.isscalar(min_bounds):
        min_bound_array = np.ones(x.shape) * min_bounds
    elif isinstance(min_bounds, np.ndarray):
        min_bound_array = min_bounds

    if np.isscalar(max_bounds):
        max_bound_array = np.ones(x.shape) * max_bounds
    elif isinstance(max_bounds, np.ndarray):
        max_bound_array = max_bounds

    best_sol = x
    best_norm = np.inf

    for n in range(num_iterations):
        gradient = compute_gradient(G, GT, d, x, verbose=(verbose == 2))
        norm = np.linalg.norm(gradient)

        if norm < best_norm:
            best_norm = norm
            best_sol = x

        if verbose in [1, 2]:
            print(n, norm)
        norms[n] = norm
        if norm <= tol:
            break

        if alpha_decay:
            alph = alpha / (n + 1)
        else:
            alph = alpha

        if grad_perturb:
            gradient *= np.random.uniform(0.0, 1.5, size=gradient.shape)
        x_new = x - (alph * norm) * gradient

        if min_bounds is not None:
            x_new = np.maximum(min_bound_array, x_new)

        if max_bounds is not None:
            x_new = np.minimum(max_bound_array, x_new)

        x = x_new

    print("norm", best_norm)

    return best_sol, norms


def solve_nnls(G, d, maxiter=None):
    x, rnorm = nnls(
        G,
        d,
        maxiter=maxiter,
    )

    print("norm", rnorm)
    return x


def solve_lsq_linear_bounded(G, d, min_bounds=None, max_bounds=None, **kwargs):
    if np.isscalar(min_bounds):
        min_bound_array = np.ones(G.shape[1]) * min_bounds
    elif isinstance(min_bounds, np.ndarray):
        min_bound_array = min_bounds

    if np.isscalar(max_bounds):
        max_bound_array = np.ones(G.shape[1]) * max_bounds
    elif isinstance(max_bounds, np.ndarray):
        max_bound_array = max_bounds

    if "bounds" in kwargs:
        bounds = kwargs.pop("bounds")
    elif min_bounds is not None and max_bounds is not None:
        bounds = list(zip(min_bound_array, max_bound_array))
    else:
        bounds = (-np.inf, np.inf)

    if "method" in kwargs:
        if kwargs["method"] == "bvls":
            if ssp.isspmatrix(G):
                G = G.todense()

    result = lsq_linear(G, d, bounds=bounds, **kwargs)

    x = result.x
    pred = result.fun

    norm = np.linalg.norm(pred - d)

    print("norm", norm)
    return x


def solve_dual_annealing(G, d, min_bounds=None, max_bounds=None, **kwargs):
    if np.isscalar(min_bounds):
        min_bound_array = np.ones(G.shape[1]) * min_bounds
    elif isinstance(min_bounds, np.ndarray):
        min_bound_array = min_bounds

    if np.isscalar(max_bounds):
        max_bound_array = np.ones(G.shape[1]) * max_bounds
    elif isinstance(max_bounds, np.ndarray):
        max_bound_array = max_bounds

    if min_bounds is not None and max_bounds is not None:
        bounds = list(zip(min_bound_array, max_bound_array))
    else:
        bounds = False

    def minimize_func(x):
        return np.linalg.norm(G.dot(x) - d)

    result = dual_annealing(minimize_func, bounds=bounds, **kwargs)

    x = result.x
    pred = result.fun

    norm = np.linalg.norm(pred - d)

    print("norm", norm)
    return x


def solve_llsq(G, d, **kwargs):
    if ssp.issparse(G):
        x = ssp.linalg.lsqr(G, d, **kwargs)[0]
        resids = G @ x - d
        norm = np.linalg.norm(resids)

    else:
        x = np.linalg.lstsq(G, d, rcond=None)[0]
        resids = G @ x - d
        norm = np.linalg.norm(resids)

    print("norm", norm)
    return x
