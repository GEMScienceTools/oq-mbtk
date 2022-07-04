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

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

FMT_BRANCH = """{0:s}<logicTreeBranch branchID="{1:s}">
{0:s}   <uncertaintyModel>{2:.3f}</uncertaintyModel>
{0:s}   <uncertaintyWeight>{3:.4f}</uncertaintyWeight>
{0:s}</logicTreeBranch>\n"""


def plot_mmax(fname, magu, pri, lkl, pos, xlim, bins, wei, wdt, sid):
    """
    Creates the plot for Mmax
    """

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 8)
    fig.suptitle(f'Source: {sid}', fontsize=16)

    # Prior
    ax1 = axs[0, 0]
    _ = ax1.plot(magu, pri)
    _ = ax1.set_ylabel('Probability')
    _ = ax1.set_xlabel('Magnitude')
    _ = ax1.set_xlim(xlim)

    # Likelihood
    ax1 = axs[0, 1]
    _ = ax1.plot(magu, lkl)
    _ = ax1.set_ylabel('Likelihood')
    _ = ax1.set_xlabel('Magnitude')
    _ = ax1.set_xlim(xlim)

    # Posterior
    ax1 = axs[1, 0]
    _ = ax1.plot(magu, pos)
    _ = ax1.set_ylabel('Probability')
    _ = ax1.set_xlabel('Magnitude')
    _ = ax1.set_xlim(xlim)

    # PMF
    ax1 = axs[1, 1]
    _ = ax1.bar(bins[:-1], wei, width=wdt, color='none', edgecolor=u'#1f77b4',
                lw=2, align='edge')
    _ = ax1.set_ylabel('Probability')
    _ = ax1.set_xlabel('Magnitude')
    _ = ax1.set_xlim(xlim)

    plt.savefig(fname)


def old_get_composite_likelihood(dfc, ccomp, bgr, last_year=None):
    """
    """
    res = 0.1
    if last_year is None:
        last_year = max(dfc.year)
    ccomp = np.array(ccomp)
    # Max observed magnitude
    mmaxobs = max(dfc.magnitude)
    # Minimum magnitude considered
    mag0 = np.floor(min(ccomp[:, 1])/res)*res
    # Maximum magnitude considered
    mag1 = np.ceil(mmaxobs/res)*res + 3.0
    mu = np.arange(mag0-1.0, mag1, 0.001)
    # Computing occurrences
    num_tot = 0
    for i, cco in enumerate(ccomp):
        up = ccomp[i-1, 0]
        if i == 0:
            up = last_year
        num = len((dfc.year > cco[0]) & (dfc.year <= up) &
                    (dfc.magnitude >= cco[1])) / (up -  cco[0])
        num_tot += num
    num_tot *= (last_year - ccomp[0, 0])
    # Likelihood
    lkl = likl(bgr, mag0, num_tot, mu, mmaxobs)
    return mu, lkl


def get_composite_likelihood(dfc, ccomp, bgr, last_year=None):
    """
    """
    res = 0.1
    # Max observed magnitude
    mmaxobs = max(dfc.magnitude)
    # Minimum magnitude considered
    mag0 = np.ceil(mmaxobs/res)*res - 3.0
    # Maximum magnitude considered
    mag1 = np.ceil(mmaxobs/res)*res + 3.0
    mu = np.arange(mag0-1.0, mag1, 0.001)
    # Computing occurrences
    num_tot = len(dfc[dfc.magnitude >= mmaxobs-1.0])
    # Likelihood
    lkl = likl(bgr, mmaxobs-1.0, num_tot, mu, mmaxobs)
    return mu, lkl


def likl(bgr, mag0, num, magu, mmaxobs):
    """
    Compute the likelihood function

    :param bgr:
        GR b-value
    :param mag0:
        Lower threshold magnitude
    :param num:
        Number of recorded earthquakes with magnitude equal or larger
        than mag0
    :param magu:
        Ipotetical mmax
    :param mmaxobs:
        Maximum magnitude observed
    """
    out = np.zeros_like(magu)
    idx = magu >= mmaxobs
    # See equation 5.2.1-1 page 5-8 in the CEUS-SSC report
    out[idx] = (1 - np.exp(-bgr*np.log(10)*(magu[idx]-mag0)))**(-num)
    return out


def get_mmax_pmf(pri_mean, pri_std, bins, **kwargs):
    """
    Computes the PMF of mmax using the methodology proposed by Johnston et al.
    (1994; vol. 1, chap 5)

    :param mmaxobs:
        Maxiumum magnitude observed
    :param mag0:
        Magnitude threshold
    :param lklhood:
        Number of earthquakes larger than mag0
    :param pri_mean:
        Prior mean magnitude
    :param pri_std:
        Prior standard deviation
    :param bgr:
        b-value of the Gutenberg-Richter relationship
    :param bins:
        Limits of the bins used to discretize the output distribution (mostly
        used for testing)
    :returns:
        A tuple with the weights and the values of magnitude (representing the
        centers of bins)
    """

    mmaxobs = kwargs.get('mmaxobs')
    lkl = kwargs.get('likelihood', None)
    mu = kwargs.get('mupp', None)
    wdt = kwargs.get('wdt', 0.5)
    bgr = kwargs.get('bgr', 1.0)
    fig_name = kwargs.get('fig_name', None)
    n_gt_n0 = kwargs.get('n_gt_n0', None)
    mag0 = kwargs.get('mag0', None)
    sid = kwargs.get('sid', 'undefined')

    # Compute likelihood distribution
    if lkl is None:
        assert mag0 is not None
        mag1 = np.min([np.ceil(mmaxobs/0.1)*0.1 + 3, 8.7])
        mu = np.arange(mag0-1.0, mag1+3, 0.001)
        lkl = likl(bgr, mag0, n_gt_n0, mu, mmaxobs)
    xlim = [min(mu), max(mu)]

    # Compute prior distribution
    pri = norm.pdf(mu, pri_mean, pri_std)

    idx = np.digitize(mu, bins)
    wei = np.zeros(len(bins)-1)
    pos = lkl*pri/(sum(lkl*pri))
    for i in np.unique(idx)[1:-1]:
        wei[i-1] = sum(pos[idx == i])
    wei = wei / sum(wei)

    # Figure
    if fig_name is not None:
        plot_mmax(fig_name, mu, pri, lkl, pos, xlim, bins, wei, wdt, sid)

    return wei, bins[:-1] + np.diff(bins)/2


def get_xml(mags, weis, sid, bsid):
    """
    Returns a string with the .xml describing the mmax uncertainty
    branch set. The ID of each branch follows the format <bset_id>_<b_id>
    where <b_id> is a integer (0 corresponds to the first branch in the logic
    tree.

    :param mags:
        A list or 1D array with the values of mmax
    :param weis:
        A list or 1D array with the weights assigned to each magnitude value
    :param sid:
        The ID of the source to which this uncertainty is applied
    :param bsid:
        The ID of the branch set
    :returns:
        A string with the .xml describing the branch set
    """

    # Branch-set definition
    spc = "   "
    ind = 2
    tmps = f"{ind*spc}<logicTreeBranchSet uncertaintyType=\"abGRAbsolute\"\n"
    tmps += f"{ind*spc}                    applyToSources=\"{sid}\"\n"
    tmps += f"{ind*spc}                    branchSetID=\"{bsid}\">\n"

    # Compute the weight for the last branch.
    rweis = np.array([float(f"{w:.4f}") for w in weis])
    rweis[-1] = 1 - np.sum(rweis[:-1])

    # Add the branches
    inda = ind + 1
    chk = 0
    cnt = 0
    for i, (mag, wei) in enumerate(zip(mags, rweis)):
        if wei < 1e-5:
            continue
        bid = f"{bsid}_{cnt:d}"
        tmps += FMT_BRANCH.format(spc*inda, bid, mag, wei)
        chk += wei
        cnt + 1
    tmps += f"{ind*spc}</logicTreeBranchSet>\n"

    # Check weights
    assert abs(1.0-chk) < 1e-5

    return tmps
