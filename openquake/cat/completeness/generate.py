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
# details=>=>.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import os
import copy
import time
import logging
from itertools import product
import toml
import numpy as np
import multiprocessing
from openquake.wkf.utils import create_folder
from openquake.cat.completeness.analysis import _make_ctab


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
    mags = np.array(config[key]['mags'], dtype=np.float32)
    years = np.array(config[key]['years'])
    num_steps = config[key].get('num_steps', 0)
    min_mag_compl = config[key].get('min_mag_compl', None)
    apriori_conditions = config[key].get('apriori_conditions', {})
    cref = config[key].get('completeness_ref', None)
    mrange = config['completeness'].get('deviation', 1.0) 
    try:
        _n_vals_per_iter = config[key]['step']
        logging.warning(
            "`step` parameter deprecated. Use `_n_vals_per_iter` instead."
        )
    except KeyError:
        _n_vals_per_iter = 8

    _get_completenesses(mags, 
                        years, 
                        folder_out=folder_out, 
                        num_steps=num_steps,
                        min_mag_compl=mini_mag_compl,
                        apriori_conditions=apriori_conditions,
                        cref=cref,
                        _n_vals_per_iter=_n_vals_per_iter,
                        mrange=mrange)


def _get_completenesses(mags, years, folder_out=None, num_steps=0,
                        min_mag_compl=None, apriori_conditions={},
                        completeness_ref=None, _n_vals_per_iter=6, mrange=1.0):
    """
    :param mags:
        A list or numpy array in increasing order
    :param years:
        A list or numpy array in increasing order
    :param folder_out:
        If not None it writes an .hdf5 file in this folder
    :param num_steps:
        The minimum number of steps in the completeness
    :param min_mag_compl:
        The minimum magnitude for which the completeness windows must be
        computed
    :param completeness_ref:
        Completness table indicating the years to be used in the possible
        tables allowed. Only magniutdes within 1.0 of the ones defined in the
        table will be included in the final set of tables. Years must be
        the same as those given in the param `years`. Default is None, which
        means no filtering of the possible tables will occur.
    :param apriori_conditions:
        A dictionary with key a value of magnitude and value a year. This
        combination of values must be included in the generated completeness
        windows
    :param _n_vals_per_iter:
        The number of values (years) to consider for each point in the
        iterations when making the possible completenesses. This is a
        parameter that controls performance, but does not change the results.
    """
    start = time.perf_counter()

    msg = 'Years must be in ascending order'
    assert np.all(np.diff(years) > 0), msg
    msg = 'Mags must be in ascending order'
    assert np.all(np.diff(mags) > 0), msg
    years = np.flipud(years)


    mags = np.asarray(mags)

    dlt = 0
    idxs = np.arange(len(mags) + dlt)
    idxs[::-1].sort()

    # Find index of the minimum magnitude of completeness
    if min_mag_compl is None:
        min_mag = min(mags)
    else:
        min_mag = min_mag_compl
    
    if len(np.where(min_mag <= mags)) < 1:
        msg = 'None of the magnitude intervals above the min_mag_compl'
        raise ValueError(msg)
    max_first_idx = np.min(np.where(min_mag <= mags))

    # Info
    print('Total number of combinations : {:,d}'.format(len(mags)**len(years)))
    print(f'Index of first magnitude     : {max_first_idx:,d}')
    print(f'Number of vals per iteration : {_n_vals_per_iter}')

    # Creating the possible completenesses
    perms = []
    for y in [years[i:min(i + _n_vals_per_iter, len(years))] 
              for i in range(0, len(years), _n_vals_per_iter)]:
        print(y)
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

    # Selecting only the completeness whose first magnitude index is lower or
    # equal than a threshold earlier max_first_idx is set to smallest magnitude
    # if no minimum is provided But also doesn't even seem to apply this
    # correctly?
    if min_mag_compl is not None:
        print("setting minimum completeness magnitude")
        perms = perms[perms[:, 0] >= max_first_idx, :]

    # Applying a-priori conditions
    for yea_str in apriori_conditions.keys():

        yea = float(yea_str)
        mag = float(apriori_conditions[yea_str])

        # Get the index of the year
        idx_yea = np.minimum(np.min(np.where(years <= yea)), len(years) - 1)

        # Get the index of the magnitude
        idx_mag = np.max(np.where(mag >= mags))

        # Keep the completeness windows that include the current apriori
        # constraint i.e. we keep all the completenesses that include this
        # magnitude-year tuple
        perms = perms[perms[:, idx_yea] >= idx_mag, :]



    if completeness_ref:
        from openquake.cat.completeness.analysis import clean_completeness
        years_ref = [c[0] for c in completeness_ref]
        mags_ref = [c[1] for c in completeness_ref]
        rem = []
        for iper, prm in enumerate(perms):

            ctab = _make_ctab(prm, years, mags)
            for yr, mg in ctab:
                if not yr in years_ref:
                    rem.append(iper)
                    continue
                index = years_ref.index(yr)
                mdiff = abs(mags_ref[index] - mg)
                if mdiff > mrange:
                    rem.append(iper)
                    continue

        perms = np.delete(perms, rem, 0)

    print(f'Total number selected        : {len(perms):,d}')

    if folder_out is not None:
        print(f'Saving completeness tables in: {folder_out:s}')
        np.save(os.path.join(folder_out, 'dispositions.npy'), perms)
        np.save(os.path.join(folder_out, 'mags.npy'), mags)
        np.save(os.path.join(folder_out, 'years.npy'), years)

    end = time.perf_counter()
    print('Time taken {}: ', start-end)
    return perms, mags, years
