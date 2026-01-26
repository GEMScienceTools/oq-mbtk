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

import logging

# from collections.abc import Mapping

import numpy as np
import scipy.sparse as ssp

import openquake.hazardlib as hz
from openquake.hazardlib.mfd.tapered_gr_mfd import mag_to_mo

from .utils import (
    SHEAR_MODULUS,
    get_mag_counts,
    get_mfd_occurrence_rates,
    rescale_mfd,
    make_rup_fault_lookup,
    get_mfd_moment,
    breakpoint,
)
from .solver import weights_from_errors


def _round_mag(mag, mag_decimals):
    if mag_decimals is None:
        return float(mag)
    return round(float(mag), mag_decimals)


def _cumulative_to_incremental_rates(occ_rates: dict) -> dict:
    mags = sorted(occ_rates.keys())
    inc = {}
    for i, mag in enumerate(mags):
        if i < len(mags) - 1:
            inc[mag] = float(occ_rates[mag]) - float(occ_rates[mags[i + 1]])
        else:
            inc[mag] = float(occ_rates[mag])
    return inc


def make_rup_rate_prior_for_fault_abs_mfd(
    fault_abs_mfd_component: dict,
    rups: list,
    mag_decimals: int | None = 1,
    cumulative: bool = False,
    default_rate: float = 0.0,
):
    """
    Build a per-rupture prior rate vector for a *single* fault/subfault MFD
    component dict (with keys 'mfd', 'rups_include', 'rup_fractions').

    Returns an array aligned with `rups_include` (same length/order).
    """
    if not isinstance(fault_abs_mfd_component, dict):
        raise TypeError("fault_abs_mfd_component must be a dict")

    mfd = fault_abs_mfd_component.get("mfd")
    rups_include = fault_abs_mfd_component.get("rups_include") or []
    rup_fractions = fault_abs_mfd_component.get("rup_fractions")

    if len(rups_include) == 0:
        return np.zeros(0, dtype=float)

    if rup_fractions is None:
        rup_fractions = [1.0] * len(rups_include)
    if len(rup_fractions) != len(rups_include):
        raise ValueError("rup_fractions must align with rups_include")

    rup_mags = np.array(
        [_round_mag(rups[i]["M"], mag_decimals) for i in rups_include],
        dtype=float,
    )
    include_w = np.asarray(rup_fractions, dtype=float)

    occ = get_mfd_occurrence_rates(
        mfd, mag_decimals=mag_decimals, cumulative=cumulative
    )
    if cumulative:
        occ = _cumulative_to_incremental_rates(occ)

    r0 = np.full(len(rups_include), float(default_rate), dtype=float)
    for mag in np.unique(rup_mags):
        bin_rate = float(occ.get(float(mag), 0.0))
        in_bin = rup_mags == mag
        w_sum = float(include_w[in_bin].sum())
        if w_sum <= 0.0:
            continue
        r0[in_bin] = bin_rate / w_sum

    return r0


def make_rup_rate_prior_from_fault_abs_mfds(
    fault_abs_mfds: dict,
    rups: list,
    mag_decimals: int | None = 1,
    cumulative: bool = False,
    default_rate: float = 0.0,
):
    """
    Build per-fault prior vectors from an outer dict keyed by fault/subfault id.

    Returns
    -------
    dict
        Mapping fault_id -> prior array aligned with that fault's `rups_include`.
    """
    priors = {}
    for fault_id, comp in (fault_abs_mfds or {}).items():
        if not isinstance(comp, dict):
            continue
        priors[fault_id] = make_rup_rate_prior_for_fault_abs_mfd(
            comp,
            rups=rups,
            mag_decimals=mag_decimals,
            cumulative=cumulative,
            default_rate=default_rate,
        )
    return priors


def make_ridge_regularization_eqns_from_fault_abs_mfds(
    fault_abs_mfds: dict,
    rups: list,
    ridge: float = 1.0,
    mag_decimals: int | None = 1,
    cumulative: bool = False,
    default_rate: float = 0.0,
    ridge_weight: float = 1.0,
):
    """
    Create a ridge (Tikhonov) regularization block derived from per-fault MFD
    components.

    For each fault/subfault, we add rows selecting just the ruptures that appear
    in that component:

        sqrt(ridge) * (r[idx] - r0_fault[idx]) ≈ 0

    so multi-fault ruptures contribute multiple ridge rows (one per fault they
    appear in), which is often what you want when trying to prevent overly-sparse
    NNLS PG solutions.
    """
    n_rups = len(rups)
    if ridge is None or ridge <= 0.0 or n_rups == 0:
        return None, None, None, None

    lhs_blocks = []
    rhs_blocks = []
    err_blocks = []
    per_fault = []
    current = 0

    priors = make_rup_rate_prior_from_fault_abs_mfds(
        fault_abs_mfds=fault_abs_mfds,
        rups=rups,
        mag_decimals=mag_decimals,
        cumulative=cumulative,
        default_rate=default_rate,
    )

    for fault_id, comp in (fault_abs_mfds or {}).items():
        if not isinstance(comp, dict):
            continue
        rups_include = comp.get("rups_include") or []
        if len(rups_include) == 0:
            continue
        include_idx = np.asarray(rups_include, dtype=int)

        # Selection matrix for this fault: one row per included rupture.
        rows = np.arange(include_idx.size, dtype=int)
        cols = include_idx
        data = np.ones(include_idx.size, dtype=float)
        lhs_f = ssp.csr_array(
            (data, (rows, cols)), shape=(include_idx.size, n_rups)
        )

        rhs_f = np.asarray(
            priors.get(fault_id, np.zeros(include_idx.size)), dtype=float
        )

        errs_f = np.full(
            include_idx.size,
            np.sqrt(float(ridge)) * float(ridge_weight),
            dtype=float,
        )

        lhs_blocks.append(lhs_f)
        rhs_blocks.append(rhs_f)
        err_blocks.append(errs_f)

        per_fault.append(
            {
                "fault_id": fault_id,
                "n_eqs": int(include_idx.size),
                "start_idx": int(current),
                "end_idx": int(current + include_idx.size),
            }
        )
        current += include_idx.size

    if not lhs_blocks:
        return None, None, None, None

    lhs = ssp.vstack(lhs_blocks).tocsr()
    rhs = np.concatenate(rhs_blocks).astype(float, copy=False)
    errs = np.concatenate(err_blocks).astype(float, copy=False)

    metadata = {
        "type": "ridge_fault_mfd_prior",
        "n_eqs": int(lhs.shape[0]),
        "details": {
            "ridge": float(ridge),
            "ridge_weight": float(ridge_weight),
            "mag_decimals": mag_decimals,
            "cumulative": cumulative,
            "default_rate": float(default_rate),
            "per_fault": per_fault,
        },
    }

    return lhs, rhs, errs, metadata


def make_slip_rate_eqns(rups, faults, seismic_slip_rate_frac=1.0):
    slip_rate_lhs = ssp.dok_array((len(faults), len(rups)), dtype=float)

    fault_ids = {fault["id"]: i for i, fault in enumerate(faults)}

    for i, rup in enumerate(rups):
        for fault_id in rup["faults"]:
            try:
                slip_rate_lhs[fault_ids[fault_id], i] = rup["D"]
            except KeyError:
                pass  # this is necessary until i figure out multi-region rups

    slip_rate_rhs = np.array([fault["slip_rate"] * 1e-3 for fault in faults])
    # slip_rate_err = np.array(
    #    [(fault["slip_rate_err"] * 1e-3) for fault in faults]
    # )
    slip_rate_err = weights_from_errors(
        [fault["slip_rate_err"] * 1e-3 for fault in faults],
        zero_error=1.0,
    )

    slip_rate_rhs *= seismic_slip_rate_frac

    # Create metadata about these equations
    eq_metadata = {
        'type': 'slip_rate',
        'n_eqs': len(faults),
        'details': {
            'fault_indices': list(range(len(faults))),
            'fault_ids': [f["id"] for f in faults],
        },
    }

    return slip_rate_lhs, slip_rate_rhs, slip_rate_err, eq_metadata


def rel_gr_mfd_rates(mags, b=1.0, a=4.0, corner_mag=None, rel=True, mfd=False):
    """
    Calculate the relative Gutenberg-Richter magnitude frequency distribution rates.

    Parameters
    ----------
    mags : list
        List of magnitudes
    b : float
        b-value
    a : float
        a-value
    rel : bool
        Whether to return relative rates
    mfd : Optional MFD
        If provided, will use this instead of the a and b values

    Returns
    -------
    dict
        Dictionary of relative rates
    """
    mags = np.sort(mags)
    rel_rates = {}

    if mfd:
        raise NotImplementedError("arbitrary MFD option not implemented")

    for i, mag in enumerate(mags):
        if not corner_mag:
            rel_rates[mag] = _get_gr_rate(mag, b, a)
        else:
            rel_rates[mag] = _get_tapered_gr_rate(mag, b, a, corner_mag)

    if rel:
        for i, mag in enumerate(mags):
            if i != 0:
                rel_rates[mag] /= rel_rates[mags[0]]
        # do this last because it's a reference for the others
        rel_rates[mags[0]] = 1.0
    return rel_rates


def _pareto(mo, corner_mo, min_mo, beta):
    return (min_mo / mo) ** beta * np.exp((min_mo - mo) / corner_mo)


def _get_gr_rate(mag, b, a):
    return 10 ** (a - b * mag)


def _get_tapered_gr_rate(mag, b, a, corner_mag, mag_lo=4.0, mag_hi=9.05):
    beta = 2.0 / 3.0 * b
    min_mo = mag_to_mo(mag_lo)
    max_mo = mag_to_mo(mag_hi)
    mag_mo = mag_to_mo(mag)
    corner_mo = mag_to_mo(corner_mag)
    scale_numerator = _pareto(mag_mo, corner_mo, min_mo, beta)
    scale_denominator = _pareto(mag_mo, max_mo, min_mo, beta)
    gr_rate = _get_gr_rate(mag, b, a)
    return gr_rate * scale_numerator / scale_denominator


def make_rel_gr_mfd_eqns(
    rups,
    b=1.0,
    rup_include_list=None,
    rup_fractions=None,
    corner_mag=None,
    weight=1.0,
):
    """
    Creates a set of equations that enforce a relative Gutenberg-Richter
    magnitude frequency distribution using cumulative rates (N(M >= m)).
    The resulting set of equations has M rows representing the number of
    unique magnitudes in the rupture set, and N columns representing each
    rupture.

    Parameters
    ----------
    rups : list of dicts

    b : float
        Gutenberg-Richter b-value

    rup_include_list : Optional list of ruptures to include in equation set

    rup_fractions : Optional list of fractions to apply to included ruptures

    corner_mag : Optional corner magnitude for tapered GR

    weight: float
        Weight to apply to the equation set

    """
    mag_counts = get_mag_counts(rups)

    unique_mags = sorted(mag_counts.keys())
    if len(unique_mags) == 1:
        return None, None, None, None

    ref_mag = unique_mags[0]

    rel_rates = rel_gr_mfd_rates(unique_mags, b, corner_mag=corner_mag)
    rel_rates_adj = {M: 1 / rel_rates[M] for M in unique_mags}

    mag_rup_idxs = {M: [] for M in unique_mags}
    mag_rup_fracs = {M: [] for M in unique_mags}

    if rup_include_list is None:
        included = [(i, rup, 1.0) for i, rup in enumerate(rups)]
    else:
        # Create a mapping from rup index to fraction
        if rup_fractions is None:
            frac_map = {idx: 1.0 for idx in rup_include_list}
        else:
            frac_map = {
                idx: frac for idx, frac in zip(rup_include_list, rup_fractions)
            }
        included = [
            (i, rup, frac_map[i])
            for i, rup in enumerate(rups)
            if i in frac_map
        ]

    for M in unique_mags:
        for i, rup, frac in included:
            if rup["M"] >= M:
                mag_rup_idxs[M].append(i)
                mag_rup_fracs[M].append(frac)

    n_eqs = len(unique_mags) - 1
    rel_mag_eqns = ssp.dok_array((n_eqs, len(rups)), dtype=float)
    for i, M in enumerate(unique_mags[1:]):
        for idx, frac in zip(mag_rup_idxs[ref_mag], mag_rup_fracs[ref_mag]):
            rel_mag_eqns[i, idx] += -rel_rates_adj[ref_mag] * frac
        for idx, frac in zip(mag_rup_idxs[M], mag_rup_fracs[M]):
            rel_mag_eqns[i, idx] += rel_rates_adj[M] * frac

    rel_mag_eqns_lhs = rel_mag_eqns
    rel_mag_eqns_rhs = np.zeros(n_eqs)
    rel_mag_eqns_errs = np.array([(rel_rates_adj[M]) for M in unique_mags[1:]])
    rel_mag_eqns_errs *= weight

    mag_pairs = [(ref_mag, M) for M in unique_mags[1:]]
    eq_metadata = {
        'type': 'mfd_rel',
        'n_eqs': n_eqs,
        'details': {
            'magnitude_pairs': mag_pairs,
            'reference_magnitude': ref_mag,
            'b_value': b,
            'corner_mag': corner_mag,
        },
    }

    return rel_mag_eqns_lhs, rel_mag_eqns_rhs, rel_mag_eqns_errs, eq_metadata


def mean_slip_rate(fault_sections: list, faults: list):
    """
    Calculate the mean slip rate of a fault from its sections

    Parameters
    ----------
    fault_sections : list
        List of fault sections
    faults : list
        List of faults

    Returns
    -------
    float
        Mean slip rate of the fault
    """

    slip_rates = []
    total_area = 0.0
    for section in fault_sections:
        f = faults[section]
        slip_rates.append(f["slip_rate"] * f["area"])
        total_area += f["area"]

    return np.sum(slip_rates) / total_area


def make_abs_mfd_eqns(
    rups,
    mfd,
    mag_decimals=1,
    rup_include_list=None,
    rup_fractions=None,
    weight=1.0,
    normalize=False,
    cumulative=False,
    region_name=None,
    min_mfd_error=1e-5,
):
    """
    Vectorized build of absolute MFD equations.
    Rows = magnitudes, Cols = ruptures.
    """

    # TODO: Cumulative fits don't work well. Need to investigate.

    # --- magnitudes present in rups and target MFD rates ---
    # (If you prefer your helper, keep it; np.unique is a drop-in speedup)
    # mag_counts = get_mag_counts(rups)  # current way
    # unique_mags = sorted(mag_counts.keys())
    M = np.array([rup["M"] for rup in rups], dtype=np.float64)
    # Bin rupture magnitudes to the same discretization used by the target MFD.
    # `get_mfd_occurrence_rates(..., mag_decimals=...)` rounds the MFD keys; if
    # rupture magnitudes are not binned similarly (or have float noise), RHS
    # lookup can spuriously return 0.0, producing extreme row weights.
    if mag_decimals is not None:
        M = np.array(
            [round(float(m), mag_decimals) for m in M], dtype=np.float64
        )
    unique_mags = np.unique(M)  # sorted ascending

    mfd_occ_rates = get_mfd_occurrence_rates(
        mfd, mag_decimals=mag_decimals, cumulative=cumulative
    )

    n_rups = M.size
    n_mags = unique_mags.size

    # --- per-rup weights: inclusion mask + optional fractions ---
    w = (
        np.zeros(n_rups, dtype=np.float64)
        if rup_include_list is not None
        else np.ones(n_rups, dtype=np.float64)
    )
    if rup_include_list is not None:
        # map selected rup index -> fraction (default 1.0)
        if rup_fractions is None:
            frac_map = {idx: 1.0 for idx in rup_include_list}
        else:
            # assume parallel arrays: rup_include_list[k] matches rup_fractions[k]
            frac_map = {
                idx: frac for idx, frac in zip(rup_include_list, rup_fractions)
            }
        # set weights for included rups
        for idx, frac in frac_map.items():
            if 0 <= idx < n_rups:
                w[idx] = frac  # 0 elsewhere (excluded)

    # --- broadcast selection matrix (n_rups x n_mags) ---
    if cumulative:
        sel = M[:, None] >= unique_mags[None, :]
    else:
        sel = M[:, None] == unique_mags[None, :]

    # coefficients: apply rup weights in one shot and transpose to (n_mags x n_rups)
    abs_mag_eqns = (sel * w[:, None]).T  # shape: (n_mags, n_rups)

    # --- RHS aligned to unique_mags ---
    mfd_abs_rhs = np.array(
        [mfd_occ_rates.get(Mi, 0.0) for Mi in unique_mags], dtype=np.float64
    )

    # --- optional normalization (geometric mean), guard zeros ---
    if normalize:
        # only positive entries contribute to geometric mean
        pos = mfd_abs_rhs > 0
        if np.any(pos):
            norm_constant = np.exp(np.mean(np.log(mfd_abs_rhs[pos])))
            if norm_constant > 0:
                mfd_abs_rhs /= norm_constant
                abs_mag_eqns /= norm_constant

    # --- errors and weights ---
    # Note: sqrt(0) -> 0; if you want to avoid zero-variance, add small epsilon.
    mfd_abs_errs = np.sqrt(mfd_abs_rhs)
    mfd_abs_errs_weighted = (
        weights_from_errors(mfd_abs_errs, min_error=min_mfd_error) * weight
    )

    abs_mag_eqns = ssp.csr_array(abs_mag_eqns)

    eq_metadata = {
        "type": "mfd_abs",
        "n_eqs": int(n_mags),
        "details": {
            "magnitudes": unique_mags.tolist(),
            "region": region_name if region_name else "global",
            "cumulative": cumulative,
            "normalized": normalize,
        },
    }

    return abs_mag_eqns, mfd_abs_rhs, mfd_abs_errs_weighted, eq_metadata


def _make_abs_mfd_eqns_old(
    rups,
    mfd,
    mag_decimals=1,
    rup_include_list=None,
    rup_fractions=None,
    weight=1.0,
    normalize=False,
    cumulative=False,
    region_name=None,
    min_mfd_error=1e-5,
):
    """
    This function is useful as a reference
    """
    mag_counts = get_mag_counts(rups)
    unique_mags = sorted(mag_counts.keys())

    mfd_occ_rates = get_mfd_occurrence_rates(
        mfd, mag_decimals=mag_decimals, cumulative=cumulative
    )

    mag_rup_idxs = {M: [] for M in unique_mags}
    mag_rup_fracs = {M: [] for M in unique_mags}

    if rup_include_list is None:
        for i, rup in enumerate(rups):
            if cumulative:
                for mag in unique_mags:
                    if rup["M"] <= mag:
                        mag_rup_idxs[mag].append(i)
            else:
                mag_rup_idxs[rup["M"]].append(i)
    else:
        for i, rup in enumerate(rups):
            if i in rup_include_list:
                if cumulative:
                    for mag in unique_mags:
                        if rup["M"] >= mag:
                            mag_rup_idxs[mag].append(i)
                            if rup_fractions is not None:
                                mag_rup_fracs[mag].append(
                                    rup_fractions[rup_include_list.index(i)]
                                )
                else:
                    mag_rup_idxs[rup["M"]].append(i)
                    if rup_fractions is not None:
                        mag_rup_fracs[rup["M"]].append(
                            rup_fractions[rup_include_list.index(i)]
                        )

    if len(unique_mags) > 2:
        abs_mag_eqns = np.vstack(
            [[np.zeros(len(rups))] for i in range(len(unique_mags))]
        )
        mfd_abs_rhs = np.zeros((len(abs_mag_eqns),))

        if rup_fractions is None:
            for i, M in enumerate(unique_mags):
                abs_mag_eqns[i, mag_rup_idxs[M]] = 1.0
                mfd_abs_rhs[i] = mfd_occ_rates.get(M, 0.0)
        else:
            for i, M in enumerate(unique_mags):
                for j, mm in enumerate(mag_rup_idxs[M]):
                    abs_mag_eqns[i, mm] = mag_rup_fracs[M][j]
                mfd_abs_rhs[i] = mfd_occ_rates.get(M, 0.0)

    if normalize:  # normalize by the geometric mean of the rates
        norm_constant = np.exp(np.mean(np.log(mfd_abs_rhs)))
        mfd_abs_rhs /= norm_constant
        abs_mag_eqns /= norm_constant

    mfd_abs_errs = np.sqrt(mfd_abs_rhs)
    mfd_abs_errs_weighted = (
        weights_from_errors(mfd_abs_errs, min_error=min_mfd_error) * weight
    )

    eq_metadata = {
        'type': 'mfd_abs',
        'n_eqs': len(unique_mags),
        'details': {
            'magnitudes': unique_mags,
            'region': region_name if region_name else 'global',
            'cumulative': cumulative,
            'normalized': normalize,
        },
    }

    return abs_mag_eqns, mfd_abs_rhs, mfd_abs_errs_weighted, eq_metadata


def make_slip_rate_smoothing_eqns(
    fault_adjacence,
    faults,
    rups=None,
    slip_rate_lhs=None,
    seismic_slip_rate_frac=1.0,
    smoothing_coeff=1.0,
    smoothing_weight=1.0,
):
    adj_pairs_done = []
    slip_rate_smoothing_eqns = []

    if slip_rate_lhs is None:
        slip_rate_lhs = make_slip_rate_eqns(
            rups, faults, seismic_slip_rate_frac=seismic_slip_rate_frac
        )[0]

    if ssp.issparse(slip_rate_lhs):
        slip_rate_lhs = slip_rate_lhs.tocsr()
    else:
        slip_rate_lhs = ssp.csr_array(slip_rate_lhs)

    for i, fault in enumerate(faults):
        if i not in fault_adjacence.keys():
            continue
        adj_faults = fault_adjacence[i]
        for adj_fault in adj_faults:
            if (i, adj_fault) in adj_pairs_done:
                continue
            sm_eqn = slip_rate_lhs[i, :] - slip_rate_lhs[adj_fault, :]
            slip_rate_smoothing_eqns.append(sm_eqn)

            adj_pairs_done.append((i, adj_fault))
            adj_pairs_done.append((adj_fault, i))

    smooth_lhs = ssp.vstack(slip_rate_smoothing_eqns)
    smooth_rhs = np.zeros(smooth_lhs.shape[0])
    smooth_errs = np.ones(smooth_lhs.shape[0]) * smoothing_weight

    return smooth_lhs, smooth_rhs, smooth_errs


def get_fault_moment(faults, shear_modulus=SHEAR_MODULUS):
    fault_moments = np.array(
        [
            fault["area"] * 1e6 * shear_modulus * fault["slip_rate"] * 1e-3
            for fault in faults
        ]
    )

    fault_moment = np.sum(fault_moments)

    return fault_moment


def get_slip_rate_fraction(faults, mfd):
    fault_moment = get_fault_moment(faults)
    mfd_moment = get_mfd_moment(mfd)

    seismic_slip_rate_frac = mfd_moment / fault_moment
    return seismic_slip_rate_frac


def make_fault_mfd_equation_components(
    fault_mfds,
    rups,
    fault_network,
    fault_key="subfaults",
    rup_key="rupture_df_keep",
    seismic_slip_rate_frac=1.0,
    skip_missing_rup_idxs=False,
    full_counting=True,
):
    """
    Construct per-element magnitude–frequency constraint components for a
    rupture-rate inversion.

    For each key in `fault_mfds` (typically a fault id or subfault id), this
    function returns:
      - the MFD object for that element (optionally rescaled by
        `seismic_slip_rate_frac`),
      - `rups_include`: indices into the provided `rups` list identifying
        ruptures that involve
        (touch) the element, based on `fault_network[rup_key]` and `fault_key`,
      - `rup_fractions`: per-rupture weights to apply in MFD equations; when
        using full-counting,
        each included rupture has weight 1.0.

    Parameters
    ----------
    fault_mfds : dict
        Mapping from element id -> MFD object.
    rups : list[dict]
        Rupture dictionaries. Each rupture must have an 'idx' used to match
        against indices
        referenced by `fault_network[rup_key]`.
    fault_network : dict
        Data structure containing a rupture table at `fault_network[rup_key]`
        that encodes which elements (given by `fault_key`) each rupture
        involves.
    fault_key : str
        Column/key name in `fault_network[rup_key]` indicating the element
        membership of each rupture (e.g., 'faults' or 'subfaults').
    rup_key : str
        Key in `fault_network` pointing to the rupture table used to build
        membership.
    seismic_slip_rate_frac : float or None
        If provided and not equal to 1.0, scales each element MFD to represent
        only a fraction of activity (e.g., seismic fraction). If None or 1.0,
        MFDs are unchanged.
    skip_missing_rup_idxs : bool or 'warn'
        Controls behavior when rupture indices referenced by the membership
        table cannot be found in `rups`. If True, silently skip. If 'warn', log
        and skip. If False, raise.
    full_counting: bool
        Considers that each rupture that involves a fault/subfault fully
        counts when considering the MFD of the fault.

    Returns
    -------
    dict
        Mapping from element id -> dict with keys:
          - 'mfd': MFD object for the element (possibly rescaled),
          - 'rups_include': list of integer indices into `rups`,
          - 'rup_fractions': list of float weights aligned with `rups_include`.
    """
    # Rescale MFDs only when explicitly requested
    if seismic_slip_rate_frac is None or seismic_slip_rate_frac == 1.0:
        fault_mfd_data = {
            k: {"mfd": v, "rups_include": [], "rup_fractions": []}
            for k, v in fault_mfds.items()
        }
    else:
        fault_mfd_data = {
            k: {
                "mfd": rescale_mfd(v, seismic_slip_rate_frac),
                "rups_include": [],
                "rup_fractions": [],
            }
            for k, v in fault_mfds.items()
        }

    # Lookup: container_id (fault/subfault id) -> list of rupture idx (original rup ids)
    rup_fault_lookup = make_rup_fault_lookup(fault_network[rup_key], fault_key)

    # Map rupture "idx" to its position in the passed-in `rups` list
    rup_id_count_lookup = {r["idx"]: i for i, r in enumerate(rups)}

    for container_id, on_container_rups in rup_fault_lookup.items():
        # Only build components for containers that actually have an MFD entry
        if container_id not in fault_mfd_data:
            continue

        for rup_idx in on_container_rups:
            try:
                j = rup_id_count_lookup[rup_idx]
                fault_mfd_data[container_id]["rups_include"].append(j)
                frac = 1.0
                # If rupture dictionaries carry fractional participation, use it
                # only when full-counting is disabled. This is important when
                # building per-(sub)fault MFD constraints: a rupture that spans
                # multiple (sub)faults should contribute proportionally rather
                # than counting fully in every container.
                if not full_counting:
                    if fault_key == "subfaults":
                        subfault_fracs = rups[j].get("subfault_fracs")
                        if isinstance(subfault_fracs, dict):
                            frac_val = subfault_fracs.get(container_id, 1.0)
                            if float(frac_val) != 1.0:
                                frac = float(frac_val)
                    elif fault_key == "faults":
                        fault_fracs = rups[j].get("faults_orig")
                        if isinstance(fault_fracs, dict):
                            frac_val = fault_fracs.get(container_id, 1.0)
                            if float(frac_val) != 1.0:
                                frac = float(frac_val)

                fault_mfd_data[container_id]["rup_fractions"].append(frac)
            except KeyError as e:
                if skip_missing_rup_idxs is True:
                    continue
                elif skip_missing_rup_idxs == "warn":
                    logging.info(
                        f"can't find rupture idx={rup_idx}, skipping..."
                    )
                    continue
                else:
                    raise e

    return fault_mfd_data


def make_fault_rel_mfd_equation_components(
    rups,
    fault_network,
    fault_key="subfaults",
    rup_key="rupture_df_keep",
    b_value=1.0,
    skip_missing_rup_idxs=False,
    full_counting=True,
):
    """
    Construct per-element rupture-inclusion lists for relative MFD constraints.

    This mirrors the rupture membership logic in
    `make_fault_mfd_equation_components`, but does not require per-element MFDs.
    The returned dict is compatible with the `regional_rel_mfds` structure used
    by `make_eqns` (i.e., it provides `rups_include` and `rup_fractions`).

    Parameters
    ----------
    rups : list[dict]
        Rupture dictionaries. Each rupture must have an 'idx' used to match
        against indices referenced by `fault_network[rup_key]`.
    fault_network : dict
        Data structure containing a rupture table at `fault_network[rup_key]`
        that encodes which elements (given by `fault_key`) each rupture
        involves.
    fault_key : str
        Column/key name in `fault_network[rup_key]` indicating the element
        membership of each rupture (e.g., 'faults' or 'subfaults').
    rup_key : str
        Key in `fault_network` pointing to the rupture table used to build
        membership.
    b_value : float or sequence
        Gutenberg-Richter b-value(s) to associate with each element. If a
        scalar, the same value is used for all elements. If a sequence, it
        must have length equal to the number of elements encountered via the
        rupture membership lookup; values are assigned in sorted element-id
        order when possible.
    skip_missing_rup_idxs : bool or 'warn'
        Controls behavior when rupture indices referenced by the membership
        table cannot be found in `rups`. If True, silently skip. If 'warn', log
        and skip. If False, raise.
    full_counting : bool
        If True, each included rupture contributes with fraction 1.0.
        If False, use per-rupture fractional participation (when available)
        and only override the default 1.0 when a stored fraction is not 1.0.

    Returns
    -------
    dict
        Mapping from element id -> dict with keys:
          - 'b_value': b-value associated with the element,
          - 'rups_include': list of integer indices into `rups`,
          - 'rup_fractions': list of float weights aligned with `rups_include`.
    """
    fault_rel_data = {}

    rup_fault_lookup = make_rup_fault_lookup(fault_network[rup_key], fault_key)
    try:
        container_ids = sorted(rup_fault_lookup.keys())
    except TypeError:
        container_ids = list(rup_fault_lookup.keys())

    # Normalize b-values to a per-container mapping.
    if b_value is None:
        b_value_map = {cid: None for cid in container_ids}
    elif np.isscalar(b_value):
        b_value_map = {cid: float(b_value) for cid in container_ids}
    elif isinstance(b_value, dict):
        b_value_map = {
            cid: float(b_value.get(cid, 1.0)) for cid in container_ids
        }
    else:
        n_containers = len(container_ids)
        if len(b_value) != n_containers:
            raise ValueError(
                f"b_value must be a scalar or a sequence of length {n_containers} "
                f"(got {len(b_value)})"
            )
        b_value_map = {
            cid: float(b_value[i]) for i, cid in enumerate(container_ids)
        }

    rup_id_count_lookup = {r["idx"]: i for i, r in enumerate(rups)}

    for container_id, on_container_rups in rup_fault_lookup.items():
        fault_rel_data[container_id] = {
            "b_value": b_value_map[container_id],
            "rups_include": [],
            "rup_fractions": [],
        }

        for rup_idx in on_container_rups:
            try:
                j = rup_id_count_lookup[rup_idx]
                fault_rel_data[container_id]["rups_include"].append(j)

                frac = 1.0
                if not full_counting:
                    if fault_key == "subfaults":
                        subfault_fracs = rups[j].get("subfault_fracs")
                        if isinstance(subfault_fracs, dict):
                            frac_val = subfault_fracs.get(container_id, 1.0)
                            if float(frac_val) != 1.0:
                                frac = float(frac_val)
                    elif fault_key == "faults":
                        fault_fracs = rups[j].get("faults_orig")
                        if isinstance(fault_fracs, dict):
                            frac_val = fault_fracs.get(container_id, 1.0)
                            if float(frac_val) != 1.0:
                                frac = float(frac_val)

                fault_rel_data[container_id]["rup_fractions"].append(frac)
            except KeyError as e:
                if skip_missing_rup_idxs is True:
                    continue
                elif skip_missing_rup_idxs == "warn":
                    logging.info(
                        f"can't find rupture idx={rup_idx}, skipping..."
                    )
                    continue
                else:
                    raise e

    return fault_rel_data


def make_eqns(
    rups,
    faults,
    mfd=None,
    slip_rate_eqns=True,
    seismic_slip_rate_frac=1.0,
    incremental_abs_mfds=True,
    cumulative_abs_mfds=False,
    mfd_rel_eqns=False,
    mfd_rel_b_val=1.0,
    mfd_rel_corner_mag=None,
    mfd_rel_weight=1.0,
    mfd_abs_weight=1.0,
    regional_abs_mfds=None,
    regional_rel_mfds=None,
    fault_abs_mfds=None,
    fault_abs_mfd_mode: str = "eqns",
    ridge: float = 1.0,
    ridge_weight: float = 1.0,
    ridge_cumulative: bool = False,
    ridge_default_rate: float = 0.0,
    fault_rel_mfds=None,
    mfd_abs_normalize=False,
    slip_rate_smoothing=False,
    fault_adjacence=None,
    slip_rate_smooth_weight=1.0,
    return_sparse=True,
    verbose=False,
    shear_modulus=SHEAR_MODULUS,
    return_metadata=False,
):
    """
    Modified to track and return equation metadata
    """
    lhs_set = []
    rhs_set = []
    err_set = []
    metadata_set = []
    current_eq_idx = 0

    fault_abs_mfd_mode = (fault_abs_mfd_mode or "eqns").lower()
    if fault_abs_mfd_mode not in {"eqns", "ridge"}:
        raise ValueError("fault_abs_mfd_mode must be 'eqns' or 'ridge'")

    if fault_abs_mfds is not None and fault_abs_mfd_mode == "eqns":
        if regional_abs_mfds is None:
            regional_abs_mfds = fault_abs_mfds
        else:
            if set(regional_abs_mfds.keys()).isdisjoint(
                set(fault_abs_mfds.keys())
            ):
                regional_abs_mfds.update(fault_abs_mfds)
            else:
                raise ValueError(
                    "regional_abs_mfds and fault_abs_mfds may not share keys"
                )

    if fault_rel_mfds is not None:
        if regional_rel_mfds is None:
            regional_rel_mfds = fault_rel_mfds
        else:
            if (
                len(set(regional_rel_mfds.keys()) & set(fault_rel_mfds.keys()))
                == 0
            ):
                regional_rel_mfds.update(fault_rel_mfds)
            else:
                raise ValueError(
                    "regional_rel_mfds and fault_rel_mfds may not share keys"
                )

    if seismic_slip_rate_frac is None and mfd is not None:
        fault_moment = get_fault_moment(faults, shear_modulus=shear_modulus)
        mfd_moment = get_mfd_moment(mfd)
        seismic_slip_rate_frac = mfd_moment / fault_moment
        logging.info(f"fault_moment: {float(fault_moment)}")
        logging.info(f"mfd_moment: {float(mfd_moment)}")
        logging.info(
            f"Setting seismic_slip_rate_frac to {float(seismic_slip_rate_frac)}"
        )
    elif seismic_slip_rate_frac is None and mfd is None:
        seismic_slip_rate_frac = 1.0
        logging.info(
            f"Setting seismic_slip_rate_frac to {float(seismic_slip_rate_frac)}"
        )

    if slip_rate_eqns is True:
        logging.info("Making slip rate eqns")
        slip_rate_result = make_slip_rate_eqns(
            rups,
            faults,
            seismic_slip_rate_frac=seismic_slip_rate_frac,
        )
        if slip_rate_result[-1] is not None:  # if metadata exists
            lhs, rhs, errs, metadata = slip_rate_result
            metadata['start_idx'] = current_eq_idx
            metadata['end_idx'] = current_eq_idx + metadata['n_eqs']
            current_eq_idx += metadata['n_eqs']

            lhs_set.append(lhs)
            rhs_set.append(rhs)
            err_set.append(errs)
            metadata_set.append(metadata)

    if mfd_rel_eqns is True:
        logging.info("Making MFD relative eqns")
        rel_result = make_rel_gr_mfd_eqns(
            rups,
            mfd_rel_b_val,
            corner_mag=mfd_rel_corner_mag,
            weight=mfd_rel_weight,
        )
        if rel_result is not None and rel_result[-1] is not None:
            lhs, rhs, errs, metadata = rel_result
            metadata['start_idx'] = current_eq_idx
            metadata['end_idx'] = current_eq_idx + metadata['n_eqs']
            current_eq_idx += metadata['n_eqs']

            lhs_set.append(lhs)
            rhs_set.append(rhs)
            err_set.append(errs)
            metadata_set.append(metadata)

    if regional_rel_mfds is not None:
        logging.info("Making regional MFD relative eqns")
        for reg, reg_mfd_data in regional_rel_mfds.items():
            # Check if reg_mfd_data is a dict and has required keys
            if not isinstance(reg_mfd_data, dict):
                logging.warning(f"Skipping region {reg}: data is not a dict")
                continue

            if ('rups_include' in reg_mfd_data) and (
                len(reg_mfd_data['rups_include']) > 0
            ):
                b_val = reg_mfd_data.get('b_value', 1.0)
                corner_mag = reg_mfd_data.get('corner_mag', None)
                weight = reg_mfd_data.get('weight', mfd_rel_weight)
                rup_fractions = reg_mfd_data.get('rup_fractions', None)

                reg_rel_result = make_rel_gr_mfd_eqns(
                    rups,
                    b=b_val,
                    rup_include_list=reg_mfd_data['rups_include'],
                    rup_fractions=rup_fractions,
                    corner_mag=corner_mag,
                    weight=weight,
                )
                if (
                    reg_rel_result is not None
                    and reg_rel_result[-1] is not None
                ):
                    lhs, rhs, errs, metadata = reg_rel_result
                    metadata['start_idx'] = current_eq_idx
                    metadata['end_idx'] = current_eq_idx + metadata['n_eqs']
                    metadata['details']['region'] = reg
                    current_eq_idx += metadata['n_eqs']

                    lhs_set.append(lhs)
                    rhs_set.append(rhs)
                    err_set.append(errs)
                    metadata_set.append(metadata)

    if mfd is not None:
        if incremental_abs_mfds:
            logging.info("Making MFD absolute eqns")
            abs_result = make_abs_mfd_eqns(
                rups,
                mfd,
                weight=mfd_abs_weight,
                normalize=mfd_abs_normalize,
            )
            if abs_result[-1] is not None:
                lhs, rhs, errs, metadata = abs_result
                metadata['start_idx'] = current_eq_idx
                metadata['end_idx'] = current_eq_idx + metadata['n_eqs']
                current_eq_idx += metadata['n_eqs']

                lhs_set.append(lhs)
                rhs_set.append(rhs)
                err_set.append(errs)
                metadata_set.append(metadata)

        if cumulative_abs_mfds:
            logging.info("Making cumulative MFD absolute eqns")
            cum_result = make_abs_mfd_eqns(
                rups,
                mfd,
                weight=mfd_abs_weight,
                normalize=mfd_abs_normalize,
                cumulative=True,
            )
            if cum_result[-1] is not None:
                lhs, rhs, errs, metadata = cum_result
                metadata['start_idx'] = current_eq_idx
                metadata['end_idx'] = current_eq_idx + metadata['n_eqs']
                current_eq_idx += metadata['n_eqs']

                lhs_set.append(lhs)
                rhs_set.append(rhs)
                err_set.append(errs)
                metadata_set.append(metadata)

    if regional_abs_mfds is not None:
        if incremental_abs_mfds:
            logging.info("Making regional MFD absolute eqns")
            for reg, reg_mfd_data in regional_abs_mfds.items():
                if ('rups_include' in reg_mfd_data.keys()) and (
                    len(reg_mfd_data['rups_include']) > 0
                ):
                    reg_result = make_abs_mfd_eqns(
                        rups,
                        reg_mfd_data["mfd"],
                        rup_include_list=reg_mfd_data["rups_include"],
                        rup_fractions=reg_mfd_data["rup_fractions"],
                        weight=mfd_abs_weight,
                        region_name=reg,
                    )
                    if reg_result[-1] is not None:
                        lhs, rhs, errs, metadata = reg_result
                        metadata['start_idx'] = current_eq_idx
                        metadata['end_idx'] = (
                            current_eq_idx + metadata['n_eqs']
                        )
                        current_eq_idx += metadata['n_eqs']

                        lhs_set.append(lhs)
                        rhs_set.append(rhs)
                        err_set.append(errs)
                        metadata_set.append(metadata)

        if cumulative_abs_mfds:
            logging.info("Making regional cumulative MFD absolute eqns")
            for reg, reg_mfd_data in regional_abs_mfds.items():
                if ('rups_include' in reg_mfd_data.keys()) and (
                    len(reg_mfd_data['rups_include']) > 0
                ):
                    reg_result = make_abs_mfd_eqns(
                        rups,
                        reg_mfd_data["mfd"],
                        rup_include_list=reg_mfd_data["rups_include"],
                        rup_fractions=reg_mfd_data["rup_fractions"],
                        weight=mfd_abs_weight,
                        cumulative=True,
                        region_name=reg,
                    )
                    if reg_result[-1] is not None:
                        lhs, rhs, errs, metadata = reg_result
                        metadata['start_idx'] = current_eq_idx
                        metadata['end_idx'] = (
                            current_eq_idx + metadata['n_eqs']
                        )
                        current_eq_idx += metadata['n_eqs']

                        lhs_set.append(lhs)
                        rhs_set.append(rhs)
                        err_set.append(errs)
                        metadata_set.append(metadata)

    if fault_abs_mfds is not None and fault_abs_mfd_mode == "ridge":
        logging.info("Making ridge regularization eqns from fault_abs_mfds")
        ridge_result = make_ridge_regularization_eqns_from_fault_abs_mfds(
            fault_abs_mfds=fault_abs_mfds,
            rups=rups,
            ridge=ridge,
            mag_decimals=1,
            cumulative=ridge_cumulative,
            default_rate=ridge_default_rate,
            ridge_weight=ridge_weight,
        )
        if ridge_result is not None and ridge_result[-1] is not None:
            lhs, rhs, errs, metadata = ridge_result
            metadata["start_idx"] = current_eq_idx
            metadata["end_idx"] = current_eq_idx + metadata["n_eqs"]
            current_eq_idx += metadata["n_eqs"]

            lhs_set.append(lhs)
            rhs_set.append(rhs)
            err_set.append(errs)
            metadata_set.append(metadata)

    if slip_rate_smoothing is True:
        raise NotImplementedError("Smoothing not implemented")
        # logging.info("Making slip rate smoothing eqns")
        # (
        #    slip_smooth_lhs,
        #    slip_smooth_rhs,
        #    slip_smooth_errs,
        # ) = make_slip_rate_smoothing_eqns(
        #    fault_adjacence,
        #    faults,
        #    rups,
        #    slip_rate_lhs=slip_rate_lhs,
        #    seismic_slip_rate_frac=seismic_slip_rate_frac,
        #    # smoothing_coeff=slip_rate_smooth_coeff,
        #    # smoothing_err=slip_rate_smooth_err,
        # )
        # lhs_set.append(slip_smooth_lhs)
        # rhs_set.append(slip_smooth_rhs)
        # err_set.append(slip_smooth_errs)

    logging.info("stacking results")
    if verbose:
        logging.info("matrix sizes:")
        [logging.info(lhs.shape) for lhs in lhs_set]

    if return_sparse:
        lhs_set = [ssp.csc_array(lhs) for lhs in lhs_set]
        lhs = ssp.vstack(lhs_set)
    else:
        lhs_set = [ssp.csc_array(lhs) for lhs in lhs_set]
        lhs_sparse = ssp.vstack(lhs_set)
        lhs = lhs_sparse.toarray()

    rhs = np.hstack(rhs_set)
    errs = np.hstack(err_set)

    if verbose:
        logging.info(f"lhs total: {lhs.shape}")

    if return_metadata:
        return lhs, rhs, errs, metadata_set
    else:
        return lhs, rhs, errs
