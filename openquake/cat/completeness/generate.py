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
import toml
import numpy as np
import multiprocessing
from itertools import product
from openquake.baselib import sap


def mm(a):
    return a


def get_completenesses(mags, years, folder_out=None, num_steps=0,
                       min_mag_compl=None, apriori_conditions={},
                       step=6, flexible=False):
    """
    :param mags:
        A list or numpy array in increasing order
    :param years:
        A list or numpy array in increasing order
    """

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

    if len(np.where(min_mag_compl <= mags)) < 1:
        msg = 'None of the magnitude intervals above the min_mag_compl'
        raise ValueError(msg)
    max_first_idx = np.min(np.where(min_mag_compl <= mags))

    # Info
    print('Total number of combinations : {:,d}'.format(len(mags)**len(years)))
    print('Index of first magnitude     : {:,d}'.format(max_first_idx))

    # Creating the possible completenesses
    perms = []
    for y in [years[i:min(i+step, len(years))] for i in range(0, len(years),
                                                              step)]:
        with multiprocessing.Pool(processes=8) as pool:
            p = pool.map(mm, product(idxs, repeat=len(y)))
            p = np.array(p)

            # Selecting combinations with increasing magnitude completeness
            # with time
            p = p[np.diff(p, axis=1).min(axis=1) >= 0, :]

            # Selecting combinations within min_mag_compl
            # if max(years) in y:
            #     p = p[p[:, 0] <= max_first_idx, :]

            # Updating
            if len(perms):
                new = []
                for x in perms:
                    for y in p:
                        new.append(list(x)+list(y))
                perms = new
            else:
                perms = p

    # Full set of possible completeness windows. Each perm in perms is a
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

    # Applying a-priori conditions
    for yea_str in apriori_conditions.keys():
        yea = float(yea_str)
        mag = float(apriori_conditions[yea_str])
        idx_yea = np.minimum(np.max(np.where(years >= yea)), len(years)-1)
        idx_mag = np.max(np.where(mag >= mags))
        # idxs = np.minimum(perms[:, idx_yea], len(mags)-1-dlt)
        # perms = perms[idxs <= idx_mag, :]
        perms = perms[perms[:, idx_yea] <= idx_mag, :]

    print('Total number selected        : {:,d}'.format(len(perms)))

    # Replacing the index of the 'buffer;' magnitude
    if flexible:
        perms = np.where(perms == len(mags), -1, perms)

    if folder_out is not None:
        print('Saving completeness tables in: {:s}'.format(folder_out))
        np.save(os.path.join(folder_out, 'dispositions.npy'), perms)
        np.save(os.path.join(folder_out, 'mags.npy'), mags)
        np.save(os.path.join(folder_out, 'years.npy'), years)

    return perms, mags, years


def main(fname_config, *, folder_out=None):

    config = toml.load(fname_config)
    key = 'completeness'
    mags = np.array(config[key]['mags'])
    years = np.array(config[key]['years'])
    num_steps = config[key].get('num_steps', 0)
    min_mag_compl = config[key].get('min_mag_compl', None)
    apriori_conditions = config[key].get('apriori_conditions', {})
    step = config[key].get('step', 8)
    flexible = config[key].get('flexible', False)

    get_completenesses(mags, years, folder_out, num_steps,
                       min_mag_compl, apriori_conditions,
                       step, flexible)


msg = 'Name of the .toml file with configuration parameters'
main.fname_config = msg
msg = 'Name of the folder where to store files'
main.folder_out = msg

if __name__ == '__main__':
    sap.run(main)