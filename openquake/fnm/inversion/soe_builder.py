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

import numpy as np
import scipy.sparse as ssp

import openquake.hazardlib as hz

from .utils import (
    SHEAR_MODULUS,
    get_mag_counts,
    get_mfd_occurrence_rates,
)
from .solver import weights_from_errors


def make_slip_rate_eqns(rups, faults, seismic_slip_rate_frac=1.0):
    slip_rate_lhs = np.zeros((len(faults), len(rups)))

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

    return slip_rate_lhs, slip_rate_rhs, slip_rate_err


def rel_gr_mfd_rates(mags, b=1.0, a=4.0, rel=True):
    mags = np.sort(mags)
    bin_widths = np.diff(mags)
    bin_widths = np.insert(bin_widths, 0, np.median(bin_widths))
    bin_widths = np.append(bin_widths, np.median(bin_widths))

    rel_rates = {}

    for i, mag in enumerate(mags):
        mag_rate = 10.0 ** (a - b * (mag - bin_widths[i] / 2.0)) - 10.0 ** (
            a - b * (mag + bin_widths[i + 1] / 2.0)
        )

        rel_rates[mag] = mag_rate

        if rel is True:
            rel_rates[mag] /= rel_rates[mags[0]]

    return rel_rates


def make_rel_gr_mfd_eqns(rups, b=1.0, rup_include_list=None, weight=1.0):
    """
    Creates a set of equations that enforce a relative Gutenberg-Richter
    magnitude frequency distribution. The resulting set of equations has
    M rows representing the number of unique magnitudes in the rupture set,
    and N columns representing each rupture.

    Parameters
    ----------
    rups : list of dicts

    b : float
        Gutenberg-Richter b-value

    rup_include_list : Optional list of ruptures to include in equation set

    weight: float
        Weight to apply to the equation set

    """

    mag_counts = get_mag_counts(rups)

    unique_mags = sorted(mag_counts.keys())
    ref_mag = unique_mags[0]

    rel_rates = rel_gr_mfd_rates(unique_mags, b)
    rel_rates_adj = {M: 1 / rel_rates[M] for M in unique_mags}

    mag_rup_idxs = {}
    for M in unique_mags:
        if rup_include_list is None:
            mag_rup_idxs[M] = [
                i for i, rup in enumerate(rups) if rup["M"] == M
            ]
        else:
            mag_rup_idxs[M] = [
                i
                for i, rup in enumerate(rups)
                if rup["M"] == M and rup["idx"] in rup_include_list
            ]

    if len(unique_mags) == 1:
        return

    elif len(unique_mags) > 2:
        rel_mag_eqns = np.vstack(
            [[np.zeros(len(rups))] for i in range(len(unique_mags) - 1)]
        )
        for i, M in enumerate(unique_mags[1:]):
            rel_mag_eqns[i, mag_rup_idxs[ref_mag]] = -rel_rates_adj[ref_mag]
            rel_mag_eqns[i, mag_rup_idxs[M]] = rel_rates_adj[M]
    else:
        rel_mag_eqns = np.zeros(len(rups))
        M = unique_mags[1]
        rel_mag_eqns[mag_rup_idxs[ref_mag]] = -rel_rates_adj[ref_mag]
        rel_mag_eqns[mag_rup_idxs[M]] = rel_rates_adj[M]

    rel_mag_eqns_lhs = rel_mag_eqns
    rel_mag_eqns_rhs = np.zeros(rel_mag_eqns_lhs.shape[0])  # flat, not column
    rel_mag_eqns_errs = np.sqrt([(rel_rates_adj[M]) for M in unique_mags[1:]])
    rel_mag_eqns_errs /= weight

    return rel_mag_eqns_lhs, rel_mag_eqns_rhs, rel_mag_eqns_errs


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


def make_frac_mfd_eqns(rups, faults, mfd, mag_decimals=1, weight=1.0):
    mag_counts = get_mag_counts(rups)
    unique_mags = sorted(mag_counts.keys())  # TODO this is not used

    mfd_occ_rates = get_mfd_occurrence_rates(mfd, mag_decimals=mag_decimals)

    weighted_slip_rates = {
        rup["idx"]: mean_slip_rate(rup["faults"], faults) for rup in rups
    }

    mfd_slip_rate_fracs = {}
    for M in mfd_occ_rates.keys():
        M_rups = [rup for rup in rups if rup["M"] == M]
        slip_rates = {
            i: weighted_slip_rates[i]
            for i in [rup["rup_id"] for rup in M_rups]
        }
        slip_rate_fracs = {
            i: wsr / sum(slip_rates.values()) for i, wsr in slip_rates.items()
        }
        mfd_slip_rate_fracs[M] = slip_rate_fracs

    lhs = ssp.eye(len(rups))
    rhs = np.ones(len(rups))

    for i, rup in enumerate(rups):
        rhs[i] = (
            mfd_occ_rates[rup["M"]] * mfd_slip_rate_fracs[rup["M"]][rup["idx"]]
        )

    return lhs, rhs


def make_abs_mfd_eqns(
    rups,
    mfd,
    mag_decimals=1,
    rup_include_list=None,
    rup_fractions=None,  # these will need to be combined with rup_include_list
    weight=1.0,
    normalize=False,
):
    mag_counts = get_mag_counts(rups)
    unique_mags = sorted(mag_counts.keys())

    mfd_occ_rates = get_mfd_occurrence_rates(mfd, mag_decimals=mag_decimals)
    mag_rup_idxs = {M: [] for M in unique_mags}
    mag_rup_fracs = {M: [] for M in unique_mags}

    if rup_include_list is None:
        for i, rup in enumerate(rups):
            mag_rup_idxs[rup["M"]].append(i)

    else:
        for i, rup in enumerate(rups):
            if i in rup_include_list:
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
                # mfd_abs_rhs[i] = mfd_occ_rates[M]
                mfd_abs_rhs[i] = mfd_occ_rates.get(M, 0.0)
    else:
        pass

    if normalize:  # normalizes by the geometric mean of the rates
        norm_constant = np.exp(np.mean(np.log(mfd_abs_rhs)))

        mfd_abs_rhs /= norm_constant
        abs_mag_eqns /= norm_constant

    mfd_abs_errs = np.sqrt(mfd_abs_rhs) * weight
    # mfd_abs_errs[mfd_abs_errs == 0.0] = 1.0

    return abs_mag_eqns, mfd_abs_rhs, mfd_abs_errs


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

    for i, fault in enumerate(faults):
        if i not in fault_adjacence.keys():
            continue
        adj_faults = fault_adjacence[i]
        for adj_fault in adj_faults:
            if (i, adj_fault) in adj_pairs_done:
                continue
            sm_eqn = slip_rate_lhs[i, :] - slip_rate_lhs[adj_fault, :]
            # sm_eqn *= smoothing_coeff
            slip_rate_smoothing_eqns.append(sm_eqn)

            adj_pairs_done.append((i, adj_fault))
            adj_pairs_done.append((adj_fault, i))

    # if len(slip_rate_smoothing_eqns) > 1:
    smooth_lhs = np.vstack(slip_rate_smoothing_eqns)
    # else:
    #     smooth_lhs = slip_rate_smoothing_eqns[0]
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


def get_mfd_moment(mfd):
    mfd_moment = sum(
        [
            hz.mfd.tapered_gr_mfd.mag_to_mo(k) * v
            for k, v in get_mfd_occurrence_rates(mfd).items()
        ]
    )
    return mfd_moment


def get_slip_rate_fraction(faults, mfd):
    fault_moment = get_fault_moment(faults)
    mfd_moment = get_mfd_moment(mfd)

    seismic_slip_rate_frac = mfd_moment / fault_moment
    return seismic_slip_rate_frac


def make_eqns(
    rups,
    faults,
    mfd=None,
    slip_rate_eqns=True,
    seismic_slip_rate_frac=1.0,
    regional_slip_rate_fracs=None,
    mfd_rel_eqns=False,
    mfd_rel_b_val=1.0,
    mfd_rel_weight=1.0,
    mfd_abs_weight=1.0,
    regional_abs_mfds=None,
    mfd_abs_normalize=False,
    slip_rate_smoothing=False,
    fault_adjacence=None,
    slip_rate_smooth_weight=1.0,
    return_sparse=True,
    verbose=False,
    shear_modulus=SHEAR_MODULUS,
):

    lhs_set = []
    rhs_set = []
    err_set = []

    if seismic_slip_rate_frac is None and mfd is not None:
        fault_moment = get_fault_moment(faults, shear_modulus=shear_modulus)
        mfd_moment = get_mfd_moment(mfd)

        seismic_slip_rate_frac = mfd_moment / fault_moment
        if verbose:
            print("fault_moment", fault_moment)
            print("mfd_moment", mfd_moment)
            print(
                "setting seismic_slip_rate_frac to: ", seismic_slip_rate_frac
            )
    elif seismic_slip_rate_frac is None and mfd is None:
        print("setting seismic_slip_rate_frac to: ", seismic_slip_rate_frac)
        seismic_slip_rate_frac = 1.0

    if slip_rate_eqns is True:
        print("Making slip rate eqns")
        slip_rate_lhs, slip_rate_rhs, slip_rate_errs = make_slip_rate_eqns(
            rups,
            faults,
            seismic_slip_rate_frac=seismic_slip_rate_frac,
        )
        lhs_set.append(slip_rate_lhs)
        rhs_set.append(slip_rate_rhs)
        err_set.append(slip_rate_errs)

    if mfd_rel_eqns is True:
        print("Making MFD relative eqns")
        (mfd_rel_lhs, mfd_rel_rhs, mfd_rel_errs) = make_rel_gr_mfd_eqns(
            rups,
            mfd_rel_b_val,
            weight=mfd_rel_weight,
        )
        lhs_set.append(mfd_rel_lhs)
        rhs_set.append(mfd_rel_rhs)
        err_set.append(mfd_rel_errs)

    if mfd is not None:
        print("Making MFD absolute eqns")
        mfd_abs_lhs, mfd_abs_rhs, mfd_abs_errs = make_abs_mfd_eqns(
            rups,
            mfd,
            weight=mfd_abs_weight,  # ^ mfd_abs_weight reduces relevance
            normalize=mfd_abs_normalize,
        )
        lhs_set.append(mfd_abs_lhs)
        rhs_set.append(mfd_abs_rhs)
        err_set.append(mfd_abs_errs)

    if regional_abs_mfds is not None:
        print("Making regional MFD absolute eqns")
        for reg, reg_mfd_data in regional_abs_mfds.items():
            if ('rups_include' in reg_mfd_data.keys()) and (
                len(reg_mfd_data['rups_include']) > 0
            ):
                reg_abs_lhs, reg_abs_rhs, reg_abs_errs = make_abs_mfd_eqns(
                    rups,
                    reg_mfd_data["mfd"],
                    rup_include_list=reg_mfd_data["rups_include"],
                    rup_fractions=reg_mfd_data["rup_fractions"],
                    weight=mfd_abs_weight,
                )
                lhs_set.append(reg_abs_lhs)
                rhs_set.append(reg_abs_rhs)
                err_set.append(reg_abs_errs)

    if slip_rate_smoothing is True:
        print("Making slip rate smoothing eqns")
        (
            slip_smooth_lhs,
            slip_smooth_rhs,
            slip_smooth_errs,
        ) = make_slip_rate_smoothing_eqns(
            fault_adjacence,
            faults,
            rups,
            slip_rate_lhs=slip_rate_lhs,
            seismic_slip_rate_frac=seismic_slip_rate_frac,
            # smoothing_coeff=slip_rate_smooth_coeff,
            # smoothing_err=slip_rate_smooth_err,
        )
        lhs_set.append(slip_smooth_lhs)
        rhs_set.append(slip_smooth_rhs)
        err_set.append(slip_smooth_errs)

    print("stacking results")
    if verbose:
        print("matrix sizes:")
        [print(lhs.shape) for lhs in lhs_set]

    if return_sparse:
        lhs_set = [ssp.csc_array(lhs) for lhs in lhs_set]
        lhs = ssp.vstack(lhs_set)
    else:
        lhs = [lhs.todense() for lhs in lhs_set if ssp.issparse(lhs)]
        lhs = np.vstack(lhs_set)

    rhs = np.hstack(rhs_set)
    errs = np.hstack(err_set)

    return (
        lhs,
        rhs,
        errs,
    )
