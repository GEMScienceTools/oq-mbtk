import time
from math import log

import numpy as np
import scipy.sparse as ssp
from numba import jit, njit, prange, float64, int64, int32
from numba.experimental import jitclass

from .fastmath import cscmatvec


from IPython.core.debugger import Pdb


@jitclass
class spmatrix:
    indptr: int32[:]
    indices: int32[:]
    data: float64[:]
    shape: int64[:]

    def __init__(self, indptr, indices, data, shape):
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.shape = np.array([shape[0], shape[1]])

    # @classmethod
    # def from_csc_matrix(cls, csc_matrix):
    #    return spmatrix(csc_matrix.indptr, csc_matrix.indices, csc_matrix.data,
    #                    csc_matrix.shape)


def csc_matrix_to_spmatrix(csc_matrix):
    return spmatrix(
        csc_matrix.indptr, csc_matrix.indices, csc_matrix.data, csc_matrix.shape
    )


@njit
def geom_mean(vals):
    return np.exp(np.mean(np.log(vals)))


@njit
def rup_rate_likelihood(preds, rhs_vec, rhs_std, like_min=1e-100):
    misfit = preds - rhs_vec
    term1 = 1 / np.sqrt(2 * np.pi * rhs_std**2)
    term2 = np.exp(-0.5 * (misfit / rhs_std) ** 2)

    likes = term1 * term2
    likes[likes < like_min] = like_min

    return geom_mean(likes)


@njit
def _eval_x(A: spmatrix, x: float64[:], d: float64[:], mult_result: float64[:]):
    # zero out mult_result just in case there are values in the wrong place
    mult_result *= 0.0

    # check data sizes
    N_col = A.shape[1]
    M_row = A.shape[0]

    if len(x) != N_col:
        raise ValueError("A has different number of columns than x")
    if len(d) != M_row:
        raise ValueError("A has different number of rows than d")
    if len(mult_result) != M_row:
        raise ValueError("A has different number of rows than mult_result")

    cscmatvec(N_col, A.indptr, A.indices, A.data, x, mult_result)

    # likelihood = rup_rate_likelihood(mult_result, d, np.mean(d))
    misfit = np.sum((mult_result - d) ** 2)

    return misfit


def sample(x, scale, T, min_bounds, max_bounds):
    pass


@njit
def sample_normal_log_space(
    log_x: float64[:],
    scale: float64,
    l_min_bounds: float64[:],
    l_max_bounds: float64[:],
    # rand_gen,
):
    # new_x = rand_gen.laplace(x, scale, size=x.shape)
    new_x = log_x + log_x * np.random.randn(len(log_x)) * scale
    # new_x = rand_gen.random(len(x)) * scale + x
    new_x = np.clip(new_x, l_min_bounds, l_max_bounds)

    return np.exp(new_x)


@njit
def sample_laplace_log_space(
    x: float64[:],
    # new_x: float64[:],
    scale: float64,
    l_min_bounds: float64[:],
    l_max_bounds: float64[:],
    replace_frac: float64 = 0.01,
    replace_num: int64 = 0,
):
    if replace_num == 0:
        replace_probs = np.random.rand(len(x))

        replace_idxs_ = []
        num_replacements = 0
        for i, p in enumerate(replace_probs):
            if p <= replace_frac:
                replace_idxs_.append(i)
                num_replacements += 1
        # print(num_replacements)
        if num_replacements > 0:
            replace_idxs = np.array(replace_idxs_)
        else:
            replace_idxs = np.random.randint(0, len(x), size=1)
    else:
        replace_idxs = np.random.randint(0, len(x), size=replace_num)

    new_x = np.zeros(len(x))
    # new_x *= 0.
    new_x += x

    # for i in [i]:
    for i in replace_idxs:
        if x[i] <= 0.0:
            lx = l_min_bounds[i]
        elif np.log(x[i]) < l_min_bounds[i]:
            lx = l_min_bounds[i]
        else:
            lx = np.log(x[i])

        new_lx = np.random.laplace(loc=lx, scale=scale)

        if new_lx < l_min_bounds[i]:
            new_lx = l_min_bounds[i]
        if new_lx > l_max_bounds[i]:
            new_lx = l_max_bounds[i]

        new_x[i] = np.exp(new_lx)

    # Pdb().set_trace()

    return new_x


@njit
def _single_thread_anneal(
    A: spmatrix,
    d: float64[:],
    x: float64[:],
    min_bounds: float64[:],
    max_bounds: float64[:],
    n_iters: int64,
    T: float64,
    T_min: float64,
    alpha: float64,
    current_misfit: float64 = -1.0,
    accept_norm: float64 = 1e-5,
    seed: int64 = 69,
    sample_scale: float = 1.0,
    replace_frac: float64 = 0.001,
    replace_num: int64 = 0,
    sample_with_T: bool = False,
):
    np.random.seed(seed)

    mult_result = d * 0.0  # preallocate memory

    if current_misfit == -1.0:
        current_misfit = _eval_x(A, x, d, mult_result)

    l_min_bounds = np.log(min_bounds)
    l_max_bounds = np.log(max_bounds)

    acceptance_rands = np.random.rand(n_iters)
    acceptance_probs = np.zeros(n_iters)

    misfits = np.ones(n_iters) * 10.0
    current_misfits = np.ones(n_iters) * 10.0
    alltime_best_x = x
    i = 0

    while T > T_min and current_misfit > accept_norm and i < n_iters:
        candidate_x = sample_laplace_log_space(
            x,
            sample_scale,
            l_min_bounds,
            l_max_bounds,
            replace_frac=replace_frac * 10 * T,
            replace_num=replace_num,
        )

        candidate_misfit = _eval_x(A, candidate_x, d, mult_result)

        misfits[i] = candidate_misfit
        misfit_diff = candidate_misfit - current_misfit

        if misfit_diff <= 0.0:
            alltime_best_x = candidate_x
            accept_prob = 1.0

        elif misfit_diff > 0.0:
            if sample_with_T:
                accept_prob = 0.0 if T == 0.0 else np.exp(-misfit_diff / T)
            else:
                accept_prob = 0.0

        acceptance_probs[i] = accept_prob

        if misfit_diff <= 0.0 or acceptance_rands[i] <= accept_prob:
            x = candidate_x
            current_misfit = candidate_misfit
        current_misfits[i] = current_misfit

        T = T * alpha
        i += 1

    return alltime_best_x, current_misfits


@njit(parallel=True)
def _parallel_anneal(
    n_threads: int64,
    A: spmatrix,
    d: float64[:],
    x: float64[:],
    min_bounds: float64[:],
    max_bounds: float64[:],
    n_iters: int64,
    T: float64,
    T_min: float64,
    alpha: float64,
    current_misfit: float64 = -1.0,
    accept_norm: float64 = 1e-5,
    seed: int64 = 69,
    sample_scale=1.0,
    replace_frac: float64 = 0.001,
    replace_num: int64 = 0,
):
    best_misfit = current_misfit

    thread_results = np.zeros((n_threads, len(x)))
    thread_misfits = np.ones((n_threads, n_iters))
    thread_final_misfits = np.ones(n_threads)

    for i in prange(n_threads):
        thread_results[i, :], thread_misfits[i, :] = _single_thread_anneal(
            A=A,
            d=d,
            x=x,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            n_iters=n_iters,
            T=T,
            T_min=T_min,
            alpha=alpha,
            current_misfit=-1.0,
            accept_norm=accept_norm,
            seed=seed + i,
            sample_scale=sample_scale,
            replace_frac=replace_frac,
            replace_num=replace_num,
        )
        thread_final_misfits[i] = thread_misfits[i, -1]

    this_best_misfit = np.min(thread_final_misfits)
    best_i = np.argmin(thread_final_misfits)
    best_x = thread_results[best_i, :]

    best_misfits = thread_misfits[best_i, :]
    best_misfit = thread_final_misfits[best_i]

    return best_x, best_misfits


@njit
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1


def simulated_annealing(
    A,
    d,
    x0,
    weights=None,
    min_bounds=1e-20,
    max_bounds=1e-2,
    initial_temp=1.0,
    T_min=0.0,
    accept_norm=np.sqrt(1e-5),
    max_iters=int(1e4),
    scale=0.1,
    parallel=False,
    n_threads=9,
    max_minutes=30.0,
    meetup_iters=10,
    seed=None,
    replace_frac: float = 0.01,
    replace_num: int = 0,
):
    t0 = time.time()

    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)

    alpha = 1 - (10 / max_iters)

    if np.isscalar(min_bounds):
        min_bounds = np.ones(x0.shape) * min_bounds

    if np.isscalar(max_bounds):
        max_bounds = np.ones(x0.shape) * max_bounds

    if weights is not None:
        Aw = ssp.csc_array(np.diag(weights)) @ A
        dw = weights * d

    else:
        Aw = A
        dw = d

    print(Aw.todense())
    print(dw)

    if ssp.isspmatrix(Aw):
        Asp = csc_matrix_to_spmatrix(ssp.csc_array(Aw))

    else:
        raise ValueError("A needs to be sparse right now")

    misfit_default = 10.0

    misfit_history = np.ones(max_iters) * misfit_default

    if parallel == False:
        x = x0
        T = initial_temp
        best_misfit = _eval_x(Asp, x0, dw, np.zeros(dw.shape))
        print("init", best_misfit)
        x, current_misfits = _single_thread_anneal(
            A=Asp,
            d=dw,
            x=x0,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            n_iters=max_iters,
            T=initial_temp,
            T_min=T_min,
            alpha=alpha,
            current_misfit=-1.0,
            accept_norm=accept_norm,
            seed=seed,
            sample_scale=scale,
            replace_frac=replace_frac,
            replace_num=replace_num,
        )
        misfit_history = current_misfits
        if current_misfits[-1] > best_misfit:
            x = x0

    else:
        # intialize
        i = 0
        x = x0
        T = initial_temp
        best_misfit = _eval_x(Asp, x0, dw, np.zeros(dw.shape))
        print("init", best_misfit)
        while (
            i < max_iters
            and best_misfit > accept_norm
            and (time.time() - t0 < max_minutes * 60.0)
        ):
            X, current_misfits = _parallel_anneal(
                n_threads=n_threads,
                A=Asp,
                d=dw,
                x=x,
                min_bounds=min_bounds,
                max_bounds=max_bounds,
                n_iters=meetup_iters,
                T=T,
                T_min=T_min,
                alpha=alpha,
                current_misfit=best_misfit,
                accept_norm=accept_norm,
                seed=seed + i,
                sample_scale=scale,
                replace_frac=replace_frac,
                replace_num=replace_num,
            )
            try:
                misfit_history[i : i + meetup_iters] = current_misfits
            except:
                n_vals_left = max_iters - i

                misfit_history[-n_vals_left:] = current_misfits[-n_vals_left:]
            T *= alpha**meetup_iters
            i += meetup_iters
            best_misfit = np.min(current_misfits)
            x = X

        misfit_history = misfit_history[misfit_history != misfit_default]

    print("best misfit", np.min(current_misfits))
    return x, misfit_history
