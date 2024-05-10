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
import pandas as pd
import matplotlib.pyplot as plt

import openquake as oq


def plot_mfd_accumdict(mfd, **kwargs):
    mags = sorted(mfd.keys())
    vals = np.cumsum(np.array([mfd[k] for k in mags])[::-1])[::-1]

    plt.semilogy(mags, vals, **kwargs)
    plt.xlabel("M")
    plt.ylabel("Cumulative number of ruptures")


def plot_mfd(mfd, errs=False, label=None, **kwargs):
    mags = [ar[0] for ar in mfd.get_annual_occurrence_rates()]
    rates = [ar[1] for ar in mfd.get_annual_occurrence_rates()]
    stds = np.sqrt(rates)

    cum_rates = np.cumsum(rates[::-1])[::-1]
    cum_stds = np.cumsum(stds[::-1])[::-1]

    if errs:
        plt.errorbar(
            mags,
            cum_rates,
            yerr=cum_stds,
            label=label,
        )

    else:
        plt.plot(mags, cum_rates, label=label, **kwargs)

    plt.yscale("log")


def plot_seis(
    eqs,
    mag_col="magMw",
    year_col="year",
    start_year=None,
    latest_year=None,
    completeness_table=None,
    **kwargs
):
    eqplot = eqs.copy(deep=True)

    if latest_year is None:
        latest_year = eqs[year_col].max()

    if start_year is None:
        start_year = eqs[year_col].min()

    if completeness_table is not None:
        cc = pd.DataFrame(
            [{"yr": c[0], "mag": c[1]} for c in completeness_table]
        )

        def get_comp_year(mag, cc=cc, small_val=-1):
            if cc.mag.min() > mag:
                comp_year = small_val
            else:
                comp_year = cc[cc.mag <= mag].yr.min()
            return comp_year

        eqplot["comp_year"] = eqplot[mag_col].apply(get_comp_year)

        mfd = oq.baselib.general.AccumDict()

        for i, rup in eqplot.iterrows():
            mfd += {rup[mag_col]: 1 / (latest_year - rup["comp_year"])}

        mags = sorted(mfd.keys())
        vals = np.cumsum(np.array([mfd[k] for k in mags])[::-1])[::-1]

    else:
        mags = np.sort(eqplot[mag_col])
        vals = np.arange(len(eqplot[mag_col]))[::-1] / (
            latest_year - start_year
        )

    plt.semilogy(
        mags,
        vals,
        "--",
        label="EQs",
        **kwargs,
    )


def plot_soln_mfd(
    soln, ruptures, label=None, rup_list_include=None, mag_key="M"
):
    mfd = oq.baselib.general.AccumDict()

    if rup_list_include is None:
        for i, rup in enumerate(ruptures):
            mfd += {rup[mag_key]: soln[i]}

    plot_mfd_accumdict(mfd, label=label)


def plot_soln_slip_rates(soln, slip_rates, lhs, errs=None, units="mm/yr"):
    if units == "mm/yr":
        coeff = 1e3
    elif units == "m/yr":
        coeff = 1.0

    pred_slip_rates = lhs.dot(soln)[: len(slip_rates)] * coeff

    plt.plot(
        [0.0, slip_rates.max() * 1.1],
        [
            0.0,
            slip_rates.max() * 1.1,
        ],
        "k-",
        lw=0.2,
    )
    if errs is not None:
        plt.errorbar(
            slip_rates,
            slip_rates,
            yerr=errs,
            fmt="k,",
            lw=0.2,
        )

    plt.plot(slip_rates, pred_slip_rates, ".")

    plt.axis("equal")
    plt.xlabel("Observed slip rate")
    plt.ylabel("Predicted slip rate")


def plot_rupture_rates_w_mags(
    soln, ruptures, logy=False, negs=True, zeros=True
):
    mags = np.array([r["M"] for r in ruptures])

    pos_rates = soln[soln > 0.0]
    pos_mags = mags[soln > 0.0]

    neg_rates = soln[soln < 0]
    neg_mags = mags[soln < 0.0]

    zero_rates = soln[soln == 0.0]
    zero_mags = mags[soln == 0.0]

    plt.plot(pos_mags, pos_rates, ".")
    if zeros:
        plt.plot(zero_mags, zero_rates, "m.")
    if negs:
        plt.plot(neg_mags, neg_rates, "r.")

    if logy:
        plt.gca().set_yscale("log")
