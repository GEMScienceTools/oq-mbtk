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
import copy
from itertools import product
import toml
import numpy as np
import multiprocessing
from openquake.wkf.utils import create_folder
from openquake.cat.completeness.analysis import clean_completeness


def get_compl(perms, mags, years, idxperm, flexible):
    """
    example: get_compl(perms, mags, years, 0, True)
    """
    if flexible:
        idx = np.where(perms[idxperm, :] == len(mags) + 1, -1, perms)
    else:
        idx = perms[idxperm, :]
    idx = idx.astype(int)
    tmp = np.array([(y, m) for y, m in zip(years, mags[idx])])
    ctab = clean_completeness(tmp)
    return ctab


def mm(a):
    return a


def get_completenesses(fname_config, folder_out):
    """
    :param fname_config:
        .toml formatted file with configuration parameters.
    :param folder_out:
        Output folder where to store results i.e. the files containing all the
        possible completeness windows admitted
    """
    create_folder(folder_out)
    config = toml.load(fname_config)
    key = 'completeness'
    mags = np.array(config[key]['mags'])
    years = np.array(config[key]['years'])
    num_steps = config[key].get('num_steps', 0)
    min_mag_compl = config[key].get('min_mag_compl', None)
    # This is left for back compatibility
    apriori_conditions_out = config[key].get('apriori_conditions', {})
    apriori_conditions_out = config[key].get('apriori_conditions_out',
                                             apriori_conditions_out)
    apriori_conditions_in = config[key].get('apriori_conditions_in', {})
    step = config[key].get('step', 8)
    flexible = config[key].get('flexible', False)

    _get_completenesses(mags, years, folder_out, num_steps, min_mag_compl,
                        apriori_conditions_out, apriori_conditions_in, step,
                        flexible)


def _get_completenesses(mags, years, folder_out=None, num_steps=0,
                        min_mag_compl=None, apriori_conditions_out={},
                        apriori_conditions_in={}, step=6, flexible=False):
    """
    This function generates a set of completeness windows starting from a
    set of magnitude and time thresholds included in the `mags` and `years`
    vectors.

    The output is a triple

    :param mags:
        A list or numpy array in increasing order
    :param years:
        A list or numpy array in increasing order
    :param folder_out:
        A string with the output folder. When `None` the code does not save
        the produced output.
    :param num_steps:
        The minimum number of steps that each completeness window must have.
        Zero corresponds to a flat completeness window.
    :param min_mag_compl:
        The minimum value of magnitude used to define the completeness. Note
        that this is also the minimum magnitude that will be used to compute
        the magnitude-frequency distribution.
    :param apriori_conditions:
        This is a dictionary where keys are years and values are magnitudes.
        Each year-magnitude combination defines points (i.e. earthquakes)
        that must be left out by the selected completeness windows.
    :param step:
        The step used to parallelize the calculation [default is 6]
    :param flexible:
        A boolean. When False the completeness window is always starting on
        the right with the smallest magnitude value
    :returns:
        A triple of :class:`numpy.ndarray` instances. The first one, `perms`,
        is an array of size <number of perms> x <number of time intervals>.
        The array contains integer indexes indicating the magnitude value
        of completeness for the corresponding time period. For example,
        if `years` is [1900, 2000], `mags` is [3, 4, 5], the row of
        perms corresponding to [0, 2], states that the catalogue is complete
        above 5 since 1900 and above 3 since 2000.
    """

    if isinstance(mags, list):
        mags = np.array(mags)
    if isinstance(years, list):
        years = np.array(years)

    msg = 'Years must be in ascending order'
    assert np.all(np.diff(years) > 0), msg
    msg = 'Mags must be in ascending order'
    assert np.all(np.diff(mags) > 0), msg

    years = np.flipud(years)

    # If flexible is true we add an additional magnitude to relax the condition
    # that the lowest time interval must contain the largest magnitude
    dlt = 1 if flexible else 0
    idxs = np.arange(len(mags)+dlt)
    idxs[::-1].sort()

    # Find index of the minimum magnitude of completeness
    if min_mag_compl is None:
        min_mag_compl = min(mags)

    # Check if the catalogue contains magnitudes above the minimum threshold
    if len(np.where(min_mag_compl <= mags)) < 1:
        msg = 'None of the magnitude intervals above the min_mag_compl'
        raise ValueError(msg)
    max_first_idx = np.min(np.where(min_mag_compl <= mags))

    # Info
    tmp = len(mags)**len(years)
    print('Total number of combinations                : {:,d}'.format(tmp))
    print(f'Index of the min magnitude of completeness  : {max_first_idx:,d}')

    # Creating the all the possible completenes windows
    perms = []
    for y in [years[i:min(i+step, len(years))] for i in range(0, len(years),
                                                              step)]:
        with multiprocessing.Pool(processes=8) as pool:
            p = pool.map(mm, product(idxs, repeat=len(y)))
            p = np.array(p)

            # Selecting combinations with increasing magnitude completeness
            # with time
            p = p[np.diff(p, axis=1).min(axis=1) >= 0, :]

            # Update the list with the permutations
            if len(perms):
                new = []
                for x in perms:
                    for y in p:
                        new.append(list(x)+list(y))
                perms = new
            else:
                perms = p

    # Full set of possible completeness windows. Each perm in `perms` is a
    # list of indexes for the magnitudes in the mags vector (in increasing
    # order), starting from the most recent time interval. So 0 in the
    # first position means the lowest value of magnitude for the most
    # recent time interval
    p = np.array(copy.copy(perms))

    # Selecting only completenesses that are decreasing within increasing
    # time.
    p = p[np.diff(p, axis=1).min(axis=1) >= -1e-10, :]
    perms = p

    # Selecting only the curves with at least X steps
    i = np.count_nonzero(np.diff(perms, axis=1) > 0, axis=1)
    perms = perms[i >= num_steps, :]

    # Selecting only the completeness whose first magnitude index is
    # lower or equal than a threshold
    perms = perms[perms[:, 0] <= max_first_idx, :]

    # Replacing the index of the 'buffer;' magnitude
    if flexible:
        perms = np.where(perms == len(mags), -1, perms)

    # Applying a-priori conditions
    for yea_str in apriori_conditions_out.keys():
        yea = float(yea_str)
        mag = float(apriori_conditions_out[yea_str])
        idx_yea = np.min(np.where(yea >= years))
        idx_mag = np.min(np.where(mag < mags))
        perms = perms[perms[:, idx_yea] >= idx_mag, :]

    # Applying a-priori conditions IN
    for yea_str in apriori_conditions_in.keys():
        yea = float(yea_str)
        mag = float(apriori_conditions_in[yea_str])
        idx_yea = np.min(np.where(yea >= years))
        idx_mag = np.max(np.where(mags > mag))
        perms = perms[perms[:, idx_yea] < idx_mag, :]

    print(f'Total number selected completeness windows  : {len(perms):,d}')

    if folder_out is not None:
        print(f'Saving completeness tables in               : {folder_out:s}')
        np.save(os.path.join(folder_out, 'dispositions.npy'), perms)
        np.save(os.path.join(folder_out, 'mags.npy'), mags)
        np.save(os.path.join(folder_out, 'years.npy'), years)

    return perms, mags, years
