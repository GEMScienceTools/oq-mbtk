# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
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

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib._color_data as mcds

from matplotlib.legend import Legend

COLORS = [mcds.XKCD_COLORS[k] for k in mcds.XKCD_COLORS]
random.seed(1)
random.shuffle(COLORS)


def get_hists(dfc: pd.DataFrame, bins: np.ndarray, agencies: list = None,
              column="magMw") -> tuple[list, list]:
    """
    :param dfc:
        A :class:`pandas.DataFrame` instance
    :param bins:
        The bin edges that will be used to plot the histogram
    :param agencies:
        A list with the IDs of the agencies to consider
    :param column:
        The name of the column with the magnitude values
    """

    # Getting the list of agencies
    if not agencies:
        agencies = get_agencies(dfc)

    # Creating the histograms
    out = []
    out_agencies = []
    for key in agencies:
        m_mw = dfc[dfc['magAgency'] == key][column]
        if len(m_mw):
            hist, _ = np.histogram(m_mw, bins=bins)
            out.append(hist)
            out_agencies.append(key)
    return out, out_agencies


def get_ranges(agencies: list, dfc: pd.DataFrame,
               mthresh: float = -10.0) -> tuple[list, list]:
    """
    :param agencies:
        A list with the IDs of the agencies
    :param dfc:
        A :class:`pandas.DataFrame` instance
    :param mthresh:
        Minimum magnitude threshold
    :returns:
        A tuple with two lists. The first list contains the first and last
        year of occurrence for earthquakes detected by an agency. The second
        list contains the number of earthquakes.
    """

    # Getting the list of agencies
    if not agencies:
        agencies = get_agencies(dfc)

    # Computing the time interval
    out = []
    num = []
    for key in agencies:
        condition = (dfc['magAgency'] == key) & (dfc['value'] > mthresh)
        ylow = np.min(dfc[condition]['year'])
        yupp = np.max(dfc[condition]['year'])
        num.append(len(dfc[condition]))
        out.append([ylow, yupp])
    return out, num


def get_agencies(dfc: pd.DataFrame) -> list:
    """
    Return a list of the agencies in the catalogue

    :param df:
        A :class:`pandas.DataFrame` instance
    :return:
        A list with the list of agencies
    """
    return list(dfc["magAgency"].unique())


def plot_time_ranges(dfc: pd.DataFrame, agencies: list = None,
                     fname: str = '/tmp/tmp.pdf', **kwargs):
    """
    Creates a plot showing the interval between the first and the last
    earthquake origin of the agencies included in the database.

    :param dfc:
        A :class:`pandas.DataFrame` instance with
    :param agencies:
        A list of agencies codes
    :param fname:
        The name of the output file
    :param kwargs:
        - mthresh: minimum magnitude threshold
        - nthresh: minimum number of events threshold
    """
    tmp = sorted(get_agencies(dfc), reverse=True)
    if not agencies:
        agencies = tmp

    # Set minimum magnitude threshold
    mthresh = kwargs.get('mthresh', -10.0)
    txt_y_shift = kwargs.get('txt_y_shift', 0.2)

    # Get time range and number of events for each agency
    yranges, num = get_ranges(agencies, dfc, mthresh)

    # Filter out the agencies without enough events
    if 'nthresh' in kwargs:
        num = np.array(num)
        idx = np.nonzero(num > kwargs['nthresh'])
        num = num[idx]
        agencies = [agencies[i] for i in idx[0]]
        yranges = [yranges[i] for i in idx[0]]

    # Compute line widths
    max_wdt = 12
    min_wdt = 3
    lws = np.array(num)/max(num) * (max_wdt-min_wdt) + min_wdt

    # Plotting
    height = kwargs.get("height", 8)

    _ = plt.figure(figsize=(10, height))
    ax = plt.subplot(1, 1, 1)
    ax.tick_params(labelsize=14)
    plt.style.use('seaborn-ticks')
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['axes.labelsize'] = 16

    for i, key in enumerate(agencies):
        if sum(np.diff(yranges[i])) > 0:
            plt.plot(yranges[i], [i, i], COLORS[i], lw=lws[i])
            tmps = f'{num[i]:d} [{yranges[i][0]}-{yranges[i][1]}]'
            plt.text(yranges[i][0], i+txt_y_shift, tmps)
        else:
            if num[i] == 0:
                continue
            plt.plot(yranges[i][1], i, 'o', COLORS[i], lw=min_wdt)
            tmps = f'{num[i]:d} [{yranges[i][1]}]'
            plt.text(yranges[i][1], i+txt_y_shift, tmps)

    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle=':')
    tmps = [' ']
    tmps.extend(agencies)
    ax.set_yticks(range(len(agencies)))
    ax.set_yticklabels(agencies)

    # Creating legend for thickness
    idx2 = np.argmin(num)
    idx1 = np.argmax(num)
    xlo = min(np.array(yranges)[:, 0])
    xup = max(np.array(yranges)[:, 1])
    xdf = xup - xlo
    fake1, = plt.plot([xlo, xlo], [0, 0], lw=max_wdt, alpha=1,
                      color=COLORS[idx1])
    fake2, = plt.plot([xlo, xlo], [0, 0], lw=min_wdt, alpha=1,
                      color=COLORS[idx2])
    labels = [f'{max(num):d}', f'{min(num):d}']
    leg = Legend(ax, [fake1, fake2], labels=labels, loc='best', frameon=True,
                 title='Number of magnitudes', fontsize='medium')
    ax.add_artist(leg)
    ax.set_xlim([xlo-xdf*0.05, xup+xdf*0.05])
    plt.xlabel('Year')
    plt.savefig(fname)

    return num


def plot_histogram(dfc, agencies=None, wdt=0.1, column="magMw",
                   fname='/tmp/tmp.pdf', **kwargs):
    """
    :param dfc:
        A :class:`pandas.DataFrame` instance
    :param agencies:
        A list of agencies codes
    :param wdt:
        A float defining the width of the bins
    :param fname:
        The name of the output file
    """

    dfc = dfc.astype({column: 'float32'})

    # Filtering
    num = len(dfc)
    dfc = dfc[np.isfinite(dfc[column])]
    fmt = "Total number of events {:d}, with finite magnitude {:d}"
    print(fmt.format(len(dfc), num))

    # Info
    print('Agencies')
    print(get_agencies(dfc))

    # Settings
    if not agencies:
        agencies = get_agencies(dfc)
        print('List of agencies plotted: ', agencies)

    # Settings plottings
    plt.style.use('seaborn-ticks')
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['axes.labelsize'] = 16

    # Data
    mw = dfc[column].values

    # Creating bins and total histogram
    mmi = np.floor(min(mw)/wdt)*wdt-wdt
    mma = np.ceil(max(mw)/wdt)*wdt+wdt
    bins = np.arange(mmi, mma, step=wdt)
    hist, _ = np.histogram(mw, bins=bins)

    # Computing the histograms
    hsts, sel_agencies = get_hists(dfc, bins, agencies, column=column)

    # Create Figure
    fig = plt.figure(figsize=(15, 8))
    ax = plt.subplot(1, 1, 1)
    ax.tick_params(labelsize=14)

    # Get the CCDF
    ccdf = np.array([sum(hist[i:]) for i in range(0, len(hist))])

    # Plotting bars of the total histogram
    plt.bar(bins[:-1]+wdt/2, hist, width=wdt*0.8, color='none',
            edgecolor='blue', align='center', lw=1, )
    #
    # Plotting the cumulative histogram
    bottom = np.zeros_like(hsts[0])
    for i, hst in enumerate(hsts):
        plt.bar(bins[:-1], hst, width=wdt*0.8, color=COLORS[i],
                edgecolor='none', align='edge', lw=1,
                bottom=bottom, label=sel_agencies[i])
        bottom += hst
    #
    # Plotting the CCDF
    plt.plot(bins[1:], ccdf, color='red',
             label='Cumulative distribution (N>m)', lw=1)
    plt.yscale('log')
    plt.xlabel('Magnitude')
    plt.ylabel('Number of magnitude values')

    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle=':')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5,
              fontsize='large')

    # Save figure
    folder = os.path.dirname(fname)
    Path(folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(fname)

    if "xlim" in kwargs:
        ax.set_xlim(kwargs["xlim"])

    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])

    print(f'Created figure: {fname:s}')
    return fig, ax
