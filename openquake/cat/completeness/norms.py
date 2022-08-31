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

import copy
import scipy
from numpy import matlib
import numpy as np


def get_completeness_matrix(tcat, ctab, mbinw, ybinw):
    """
    :param tcat:
    :param ctab:
    :param mbinw:
    :param ybinw:
    """

    # Flipping ctab to make it compatible with the algorithm implemented here
    ctab = np.flipud(ctab)

    # Create the array with magnitude bins
    cat_yea = tcat.data['year']
    cat_mag = tcat.data['magnitude']
    mdlt = mbinw * 0.1
    ydlt = ybinw * 0.1

    min_m = np.floor((np.min(cat_mag)+mdlt)/mbinw)*mbinw
    max_m = np.ceil((np.max(cat_mag)+mdlt)/mbinw+1)*mbinw
    mags = np.arange(min_m, max_m, mbinw)
    cmags = mags[:-1]+mbinw/2

    min_y = np.floor((np.min(cat_yea)+ydlt)/ybinw)*ybinw
    max_y = np.ceil((np.max(cat_yea)+ydlt)/ybinw+1)*ybinw
    yeas = np.arange(min_y, max_y, ybinw)
    cyeas = yeas[:-1]+ybinw/2

    his, _, _ = np.histogram2d(cat_mag, cat_yea, bins=[mags, yeas])
    oin = copy.copy(his)
    out = copy.copy(his)

    for i, yea in enumerate(cyeas):

        if yea < ctab[0, 0]:
            oin[0:, i] = -1.0
            continue

        # Get the index of the row in the completeness table
        yidx = np.digitize(yea, ctab[:, 0])-1

        # Get the indexes of the magnitude
        midx = np.digitize(ctab[yidx, 1], mags)-1
        oin[:midx, i] = -1.0
        out[midx:, i] = -1.0

    return oin, out, cmags, cyeas


def get_norm_optimize(aval, bval, ctab, cmag, t_per, n_obs, last_year,
                      info=False):
    """
    :param aval:
    :param bval:
    :param ctab:
    :param cmag:
        An array with the magnitude values at the center of each occurrence
        bins
    :param t_per:
    :param n_obs:
    :param last_year:
    :param info:
    """

    occ = np.zeros((ctab.shape[0]))
    dur = np.zeros((ctab.shape[0]))
    mags = np.array(list(ctab[:, 1])+[10])

    for i, mag in enumerate(mags[:-1]):
        idx = (cmag >= mags[i]) & (cmag < mags[i+1])
        if np.any(idx):
            occ[i] = np.sum(n_obs[idx])
            dur[i] = t_per[np.min(np.nonzero(idx))]
        else:
            occ[i] = 0
            dur[i] = (last_year-ctab[i, 0])

    # Rates of occurrence in each magnitude bin
    rates = (10**(-bval * mags[:-1] + aval) -
             10**(-bval * mags[1:] + aval)) * dur

    # Standard deviation of the poisson model
    stds_poisson = scipy.stats.poisson.std(rates)
    idx = stds_poisson > 0
    if not np.any(idx):
        return None

    if info:
        print('stds', stds_poisson)

    # Widths of the magnitude intervals
    wdts = np.diff(mags)
    mag_diff = (mags[-1] - mags[0])

    # Difference between modelled and observed occurrences
    occ_diff = (np.abs(rates[idx] - occ[idx]))**0.5 / dur[idx]**0.5

    # Checking weights
    msg = '{:f} ≠ {:f}'.format(np.sum(wdts[idx] / mag_diff), 1.)
    if np.abs(np.sum(wdts[idx] / mag_diff) - 1.0) > 1e-5:
        raise ValueError(msg)

    if info:
        print('rates, occ', rates, occ)
        print('diff:', occ_diff)

    # This is the norm. It's the weighted sum of the normalised difference
    # between the number of computed and observed occurrences in each
    # completeness bin. We divide the sum by the number of completeness
    # intervals in order to use a measure that's per-interval i.e.
    # independent from the number of completeness intervals used
    norm = np.mean(occ_diff * wdts[idx]) / mag_diff
    norm /= np.sum(idx)

    return norm


def get_norm_optimize_a(aval, bval, ctab, cmag, t_per, n_obs,
                        binw, info=False):
    """
    Computes a norm

    :param aval:
        GR a-value
    :param bval:
        GR b-value
    :param ctab:
        Completeness table
    :param cmag:
        An array with the magnitude values at the center of each occurrence
        bins
    :param t_per:
    :param n_obs:
    :param last_year:
    :param info:
    """

    # Rates of occurrence in each magnitude bin in the completeness interval
    rates = (10**(-bval * (cmag - binw/2) + aval) -
             10**(-bval * (cmag + binw/2) + aval)) * t_per

    # Probability of observing n-occurrences in each magnitude bin
    occ_prob = np.ones_like(rates) * 0.999999
    num = (rates)**n_obs * np.exp(-rates)
    occ_prob = num / scipy.special.factorial(n_obs)
    norm = 1. - np.prod(occ_prob)

    return norm


def get_norm_optimize_b(aval, bval, ctab, tcat, mbinw, ybinw, back=5, mmin=-1,
                        mmax=10):
    """
    Computes a norm

    :param aval:
        GR a-value
    :param bval:
        GR b-value
    :param ctab:
        Completeness table
    :param tcat:
        A catalogue instance
    :param mbinw:
        Magnitude bin width
    :param ybinw:
        Time (i.e. year) bin width
    :param back:
        Delta year
    :param mmin:
        Minimum magnitude
    :param mmax:
        Maximum magnitude
    """

    # oin and out have shape mags x years
    oin, out, cmags, cyeas = get_completeness_matrix(tcat, ctab, mbinw, ybinw)

    # Selecting the rates for the magnitudes between mmin and mmax
    idx = (cmags >= mmin) & (cmags <= mmax)
    cmags = cmags[idx]
    oin = oin[idx, :]
    out = out[idx, :]

    # Rates for each magnitude bin
    rates = 10**(aval-bval*cmags-mbinw/2) + 10**(aval-bval*cmags+mbinw/2)*ybinw

    # Assuming a Poisson process, compute the standard deviation of the rates
    stds_poi = scipy.stats.poisson.std(rates)

    # Preparing matrices
    rates = matlib.repmat(np.expand_dims(rates, 1), 1, len(cyeas))
    stds_poi = matlib.repmat(np.expand_dims(stds_poi, 1), 1, len(cyeas))

    # Compute the year from when to count the occurrences
    mag_bins = cmags-mbinw/2
    mag_bins = np.append(mag_bins, cmags[-1]-mbinw/2)
    tmp = np.digitize(ctab[:, 1], mag_bins) - 1 - back
    tmp = np.maximum(np.zeros_like(tmp), tmp)

    count_in = np.abs(oin / ybinw - rates)
    idx = oin < 0
    count_in[idx] = 0

    count_out = np.abs(out / ybinw - rates)
    idx = out < 0
    count_out[idx] = 0

    norm = np.sum(count_in) - np.sum(count_out)
    return norm
