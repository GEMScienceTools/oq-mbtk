import itertools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np
from numba import njit, prange, types
from numba.typed import List

from scipy.sparse import csr_matrix

# ──────────────────────────────────────────────────────────────────────────
#  Pool initialisation  ────────────────────────────────────────────────────
#  (Executed ONCE in every worker process.)
# ──────────────────────────────────────────────────────────────────────────
def _init_pool(adj_dict, group_lookup, max_vertices):
    global ADJ_DICT, GROUP_OF, MAX_VERTS
    ADJ_DICT = adj_dict  # read‑only in workers
    GROUP_OF = group_lookup  # read‑only in workers
    MAX_VERTS = max_vertices


# ──────────────────────────────────────────────────────────────────────────
#  Per‑vertex DFS task  ────────────────────────────────────────────────────
#  (Executed MANY times across the worker pool.)
# ──────────────────────────────────────────────────────────────────────────
def _dfs_from_vertex(start_vertex: int) -> Set[frozenset[int]]:
    """
    Enumerate every connected vertex set (size 2 … MAX_VERTS) that
    - contains `start_vertex`
    - has no two vertices from the same group.
    """
    output: Set[frozenset[int]] = set()

    # stack items: (current_vertex, current_set, used_groups)
    stack = [(start_vertex, {start_vertex}, {GROUP_OF[start_vertex]})]

    while stack:
        v, current_set, used_groups = stack.pop()

        if 1 < len(current_set) <= MAX_VERTS:
            output.add(frozenset(current_set))

        if len(current_set) == MAX_VERTS:
            continue

        for nbr in ADJ_DICT[v]:
            if nbr in current_set:
                continue
            g = GROUP_OF[nbr]
            if g in used_groups:
                continue
            stack.append(
                (
                    nbr,
                    current_set | {nbr},
                    used_groups | {g},
                )
            )

    return output


# ──────────────────────────────────────────────────────────────────────────
#  Public API  ─────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
def find_connected_subsets_parallel_py(
    adj_dict: Dict[int, Sequence[int]],
    group_of: Mapping[int, int],
    max_vertices: int = 10,
    workers: int | None = None,
) -> List[List[int]]:
    """
    Parallel version of `find_connected_subsets_dfs` that enumerates
    connected subsets while guaranteeing **“one vertex per group”**.

    Parameters
    ----------
    adj_dict
        Undirected adjacency list.
    group_of
        vertex → group lookup (``dict`` or 1‑D NumPy array both OK).
    max_vertices
        Largest subset size returned.
    workers
        Number of worker processes (`None` → `cpu_count()`).

    Returns
    -------
    list[list[int]]
        All unique subsets, each converted back to a (mutable) list.
    """
    workers = workers or cpu_count()

    # ①  spin up the pool; every worker loads a local copy of the graph once
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_pool,
        initargs=(adj_dict, group_of, max_vertices),
    ) as pool:
        # ②  schedule one DFS task per start vertex
        results = pool.map(_dfs_from_vertex, adj_dict.keys(), chunksize=1)

        # ③  union everything that came back
        all_subsets: Set[frozenset[int]] = set().union(*results)

    # ④  cast back to the caller’s preferred data structure
    return [list(s) for s in all_subsets]



def find_connected_subsets_parallel(
        adj_csr: csr_matrix,
        group_lookup: Mapping[int,int],
        max_vertices: int=10,
        ):
    """
    Parameters
    ----------
    adj_csr: 
        **Undirected** adjacency matrix.
    group_lookup : 1‑D array‑like or dict
        vertex → *original* group id.  Will be re‑mapped to 0 … g‑1.
    max_vertices : int
        Largest subset size.

    Returns
    -------
    list[list[int]]
        All unique connected subsets satisfying "one vertex per group".
    """
    adj_csr = csr_matrix(adj_csr)
    indptr, indices = adj_csr.indptr, adj_csr.indices
    group_of, n_groups = compress_groups(group_lookup)

    # --- run the compiled routine -------------------------------------
    raw = find_connected_subsets_numba(indptr, indices,
                                       group_of, n_groups,
                                       max_vertices=max_vertices)

    # --- deduplicate across different start vertices ------------------
    seen = set()
    uniques = []
    for arr in raw:
        tpl = tuple(arr)          # arr is already sorted
        if tpl not in seen:
            seen.add(tpl)
            uniques.append(list(arr))

    return uniques


def compress_groups(group_lookup: dict[int, int] | list[int] | np.ndarray):
    """
    Make the group ids dense (0 … n_groups‑1) so we can index a Boolean
    array of length n_groups inside Numba.
    """
    if isinstance(group_lookup, dict):
        if not group_lookup:
            return np.array([], dtype=np.int32), 0
        
        max_vertex = max(group_lookup.keys())
        group_arr = np.empty(max_vertex + 1, dtype=np.int64)
        
        # Fill with a unique group for each missing vertex to avoid conflicts
        for i in range(len(group_arr)):
            group_arr[i] = i + max(group_lookup.values()) + 1
        
        for i, group in group_lookup.items():
            group_arr[i] = group
    else:
        group_arr = np.asarray(group_lookup, dtype=np.int64)
    
    unique, inv = np.unique(group_arr, return_inverse=True)
    return inv.astype(np.int32), int(unique.size)


@njit
def _dfs_recursive(indptr, indices,
                   group_of,
                   max_vertices,
                   current_set, set_len,
                   used_groups,
                   results):
    """
    Key change: explore neighbors of ALL vertices in current_set,
    not just the last added vertex.
    
    * current_set : pre‑allocated int32[ max_vertices ]
      (filled up to set_len – **not** Python list!)
    * used_groups : bool[ n_groups ] – modified in‑place
    * results     : numba.typed.List[ numba.types.int32[:] ]
    """
    if 1 < set_len <= max_vertices:
        # copy the slice 0:set_len into a new 1‑D array and store it
        subset = np.empty(set_len, dtype=np.int32)
        subset[:] = current_set[:set_len]
        subset.sort()             # canonical order for later dedup
        results.append(subset)

    if set_len == max_vertices:
        return

    # CRITICAL FIX: Try expanding from ANY vertex in current_set
    for idx in range(set_len):
        v = current_set[idx]
        row_start = indptr[v]
        row_end   = indptr[v + 1]
        
        for k in range(row_start, row_end):
            nbr = indices[k]
            
            # Already in path?
            in_path = False
            for i in range(set_len):
                if current_set[i] == nbr:
                    in_path = True
                    break
            if in_path:
                continue

            g = group_of[nbr]
            if used_groups[g]:
                continue

            # PUSH
            current_set[set_len] = nbr
            used_groups[g] = True

            _dfs_recursive(indptr, indices,
                           group_of,
                           max_vertices,
                           current_set, set_len + 1,
                           used_groups,
                           results)

            # POP
            used_groups[g] = False


@njit
def _enumerate_from_vertex(start_vertex,
                           indptr, indices,
                           group_of,
                           max_vertices,
                           n_groups):
    """
    Returns a typed.List[np.ndarray] of subsets that *include*
    start_vertex.  No duplicates inside.
    """
    used_groups = np.zeros(n_groups, dtype=np.bool_)
    used_groups[group_of[start_vertex]] = True

    current_set = np.empty(max_vertices, dtype=np.int32)
    current_set[0] = start_vertex

    out = List.empty_list(types.int32[:])
    _dfs_recursive(indptr, indices,
                   group_of,
                   max_vertices,
                   current_set, 1,
                   used_groups,
                   out)
    return out


@njit(parallel=True)
def find_connected_subsets_numba(indptr, indices,
                                 group_of, n_groups,
                                 max_vertices=10):
    n_vertices = group_of.shape[0]

    # Using a list of lists to avoid race conditions
    thread_results = List()
    for _ in range(n_vertices):
        thread_results.append(List.empty_list(types.int32[:]))

    # Parallel loop
    for v in prange(n_vertices):
        local_res = _enumerate_from_vertex(v, indptr, indices,
                                           group_of, max_vertices,
                                           n_groups)
        thread_results[v] = local_res

    # Concatenate all results
    master = List.empty_list(types.int32[:])
    for thread_res in thread_results:
        for subset in thread_res:
            master.append(subset)

    return master
