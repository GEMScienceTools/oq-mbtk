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
import glob
import logging
import warnings
import toml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openquake.cat.completeness.norms import (get_norm_optimize,
                                              get_norm_optimize_b,
                                              get_norm_optimize_a,
                                              get_norm_optimize_c,
                                              get_norm_optimize_d,
                                              get_norm_optimize_weichert,
                                              get_norm_optimize_gft)
from openquake.wkf.utils import _get_src_id, create_folder, get_list
from openquake.wkf.compute_gr_params import (get_weichert_confidence_intervals,
                                             _weichert_plot)
from openquake.mbt.tools.model_building.plt_tools import _load_catalogue
from openquake.mbt.tools.model_building.dclustering import _add_defaults
from openquake.hmtk.seismicity.occurrence.utils import get_completeness_counts
from openquake.hmtk.seismicity.occurrence.weichert import Weichert

warnings.filterwarnings("ignore")

MAXIMISE = ['optimize_a', 'optimize_b', 'optimize_c', 'optimize_d', 'optimize_weichert', 'optimize_gft', 'poisson']

def get_earliest_year_with_n_occurrences(ctab, cat, occ_threshold=2):
    """
    For each completeness interval, computes the year since when at least a
    number of earthquakes larger than `occ_threshold` took place.

    :param ctab:
        Completeness table
    :param cat:
        A catalogue instance
    :param occ_threshold:
        A scalar representing a number of earthquakes
    """
    c_yea = cat.data['year']
    c_mag = cat.data['magnitude']

    low_yea = []
    for i, com in enumerate(ctab):

        if i == len(ctab) - 1:
            uppmag = 10.0
        else:
            uppmag = ctab[i + 1][1]

        idx = (c_mag >= com[1]) & (c_mag < uppmag)
        years = c_yea[idx]

        if len(years) >= occ_threshold:
            low_yea.append(np.sort(years)[occ_threshold - 1])
        else:
            low_yea.append(np.NaN)

    """
    # Find the index of the completeness bin foreach magnitude
    idx = np.digitize(c_mag, bins=mags, right=False)

    # Process
    for i in range(ctab.shape[0]):
        years = c_yea[idx == i]
        if len(years) >= occ_threshold:
            low_yea.append(np.sort(years)[occ_threshold-1])
        else:
            low_yea.append(np.NaN)
    """

    return np.array(low_yea)


def clean_completeness(tmp):
    """
    The completeness table that must be simplified
    # should remove magnitudes < minmag, years </> catalaogue years

    :param tmp:
        An instance of a :class:`numpy.ndarrray`
    :returns:
        A simplified version of the initial completeness table
    """
    ctab = []

    # Check that years are in decreasing order
    msg = 'Years must be in decreasing order'
    assert np.all(np.diff(tmp[:, 0]) <= 0), msg

    # Loop on unique values of magnitude
    for m in np.unique(tmp[:, 1]):
        idx = np.nonzero(tmp[:, 1] == m)[0]
        ctab.append([tmp[max(idx), 0], m])
    ctab = np.array(ctab)
    return ctab


def check_criterion(criterion, rate, previous_norm, tvars):
    """
    Given a criterion, it computes the norm (i.e. distance between model and
    observations).

    :param criterion:
        Logic used to compute the norm
    :param rate:
        Earthquake rate value
    :param previous_norm:
        Current norm
    :param tvars:

    :returns:
        A tuple with a boolean (True when the new norm is better than the
        previous one) a rate (not always computed) and the current value
        of the norm
    """
    check = False

    binw = tvars['binw']
    bval = tvars['bval']
    aval = tvars['aval']
    ref_mag = tvars['ref_mag']
    ref_upp_mag = tvars['ref_upp_mag']
    bgrlim = tvars['bgrlim']
    ctab = tvars['ctab']
    tcat = tvars['tcat']
    last_year = tvars['last_year']
    n_obs = tvars['n_obs']
    cmag = tvars['cmag']
    t_per = tvars['t_per']

    norm = None
    if criterion == 'largest_rate':

        # Computes the rate to be maximised
        tmp_rate = 10**(-bval * ref_mag + aval)
        if ref_upp_mag is not None:
            tmp_rate -= 10**(-bval * ref_upp_mag + aval)
        norm = 1. / abs(tmp_rate)

    elif criterion == 'match_rate':

        # Computes the rate to match
        rate_ma = tvars['rate_to_match']
        tmp_rate = 10**(-bval * ref_mag + aval)
        if ref_upp_mag is not None:
            mmax_tmp = tvars['mmax_within_range']
            tmp_rate -= 10**(-bval * mmax_tmp + aval)
        norm = abs(tmp_rate - rate_ma)

    elif criterion == 'optimize':
        tmp_rate = -1
        norm = get_norm_optimize(tcat, aval, bval, ctab, cmag, n_obs, t_per, info=False)

    elif criterion == 'optimize_a':
        tmp_rate = -1
        norm = get_norm_optimize_a(aval, bval, ctab, binw, cmag, n_obs, t_per)

    elif criterion == 'optimize_b':
        tmp_rate = -1
        norm = get_norm_optimize_b(aval, bval, ctab, tcat, binw, ybinw=10.,
                                   mmin=ref_mag, mmax=ref_upp_mag)
    elif criterion == 'optimize_c':
        tmp_rate = -1
        norm = get_norm_optimize_c(tcat, aval, bval, ctab, last_year)

    elif criterion == 'gft':
        tmp_rate = -1
        norm = get_norm_optimize_gft(tcat, aval, bval, ctab, cmag, n_obs,
                                     t_per, last_year)

    elif criterion == 'weichert':
        tmp_rate = -1
        norm = get_norm_optimize_weichert(tcat, aval, bval, ctab, last_year)

    elif criterion == 'poisson':
        tmp_rate = -1
        norm = get_norm_optimize_c(tcat, aval, bval, ctab, last_year, ref_mag)

    if norm is None or np.isnan(norm):
        return False, -1, previous_norm

    # for maximise criteria, assume norm wants to be larger than prev norm 
    if criterion in MAXIMISE:
        if previous_norm < norm and bval <= bgrlim[1] and bval >= bgrlim[0]:
            check = True

    # for any other criteria, assume norm wants to be smaller than prev norm 
    elif previous_norm > norm and bval <= bgrlim[1] and bval >= bgrlim[0]:
        check = True

    return check, tmp_rate, norm


def _make_ctab(prm, years, mags):

    tmp = []
    for yea, j in zip(years, prm):
        if j >= -1e-10:
            tmp.append([yea, mags[int(j)]])
    tmp = np.array(tmp)
    if len(tmp) > 0:
        return clean_completeness(tmp)
    else:
        return 'skip'


def _completeness_analysis(fname, years, mags, binw, ref_mag, ref_upp_mag,
                           bgrlim, criterion, compl_tables, src_id=None,
                           folder_out_figs=None, rewrite=False,
                           folder_out=None):
    """
    :param fname:
        Name of the file with the catalogue
    :param years:
        Years (sorted descending)
    :param mags:
        Magnitudes
    :param ref_mag:
        The reference magnitude used to compute the rate and select the
        completeness table
    :param ref_upp_mag:
        The reference upper magnitude limit used to compute the rate and
        select the completeness table
    :param bgrlim:
        A list with lower and upper limits of the GR b-value
    :param criterion:
        The criterion used to compute the norm
    :param compl_tables:
        The set of completeness tables to be used
    :param src_id:
        A string with the source ID
    :param rewrite:
        Boolean
    """

    # Checking input
    if criterion not in ['match_rate', 'largest_rate', 'optimize', 'weichert',
                         'poisson', 'optimize_a', 'optimize_b', 'optimize_d']:
        raise ValueError('Unknown optimization criterion')

    tcat = _load_catalogue(fname)
    tcat = _add_defaults(tcat)
    tcat.data["dtime"] = tcat.get_decimal_time()

    # Info
    # Should have option to specify a mag_low != ref_mag
    mag_low = ref_mag
    idx = tcat.data["magnitude"] >= mag_low
    fmt = 'Catalogue contains {:d} events equal or above {:.1f}'
    print('\nSOURCE:', src_id)
    print(fmt.format(sum(idx), mag_low))

    # Loading all the completeness tables to be considered in the analysis
    # See http://shorturl.at/adsvA
    perms = compl_tables['perms']

    # Configuration parameters for the Weichert method
    wei_conf = {'magnitude_interval': binw,
                'reference_magnitude': 0.0,
                'bvalue': 1.0}
    weichert = Weichert()

    # Initial settings
    if criterion in MAXIMISE:
        norm = -1e1000
    else:
        norm = 1
    print("starting norm = ", norm)

    rate = -1e10
    save = []
    wei = None
    count = {'complete': 0, 'warning': 0, 'else': 0, 'early': 0}

    all_res, all_mags, all_rates = [], [], []
    # For each permuation of completeness windows, check compatability
    for iper, prm in enumerate(perms):
        tnorm = norm

        # Info
        print(f'Iteration: {iper:05d} norm: {norm:12.6e}', end="\r")

        ctab = _make_ctab(prm, years, mags)
        if isinstance(ctab, str):
            continue

        #tmp = []
        #for yea, j in zip(years, prm):
        #    if j >= -1e-10:
        #        tmp.append([yea, mags[int(j)]])
        #tmp = np.array(tmp)

        #if len(tmp) > 0:
        #    ctab = clean_completeness(tmp)
        #else:
        #    continue

        # Check compatibility between catalogue and completeness table. This
        # function finds in each magnitude interval defined in the completeness
        # table the earliest year since the occurrence of a number X of
        # earthquakes. This ensures that the completeness table applies only to
        # sets with a number of occurrences sufficient to infer a recurrence
        # interval.

        # Check that the selected completeness window has decreasing years and
        # increasing magnitudes
        assert np.all(np.diff(ctab[:, 0]) <= 0)
        assert np.all(np.diff(ctab[:, 1]) >= 0)

        # Compute occurrence

        cent_mag, t_per, n_obs = get_completeness_counts(tcat, ctab, binw)
        if len(cent_mag) == 0:
            continue
        wei_conf['reference_magnitude'] = min(ctab[:, 1])

        try:
            # Calculate weichert a and b parameters given the current
            # completeness
            bval, sigb, rmag_rate, rmag_sigma_rate, aval, siga = \
                    weichert._calculate(tcat, wei_conf, ctab)
        except:
            n_obs = [0]
            count['else'] += 1
            continue

        if np.count_nonzero(n_obs) == 0:
            count['else'] += 1
            continue

        if bval >= bgrlim[1] or bval <= bgrlim[0]:
            count['else'] += 1
            continue

        r_mag = np.floor((ref_mag + binw * 0.01) / binw) * binw - binw / 2
        r_upp_mag = (
            np.ceil((ref_upp_mag + binw * 0.01) / binw) * binw + binw / 2)

        # Create a dictionary of parameters for the function that computes
        # the norm
        tvars = {}
        tvars['binw'] = binw
        tvars['last_year'] = tcat.end_year
        tvars['bval'] = bval
        tvars['aval'] = aval
        tvars['ref_mag'] = r_mag
        tvars['ref_upp_mag'] = r_upp_mag
        tvars['bgrlim'] = bgrlim
        idx_mags = (cent_mag >= ref_mag) & (cent_mag < ref_upp_mag)
        tvars['rate_to_match'] = np.sum(n_obs[idx_mags] / t_per[idx_mags])
        idx_obs = (idx_mags) & (n_obs > 0)
        if len(idx_obs) <= 10:
            continue
        elif len(idx_obs) > len(cent_mag):
            continue

        tvars['mmax_within_range'] = np.max(cent_mag[idx_obs])
        tvars['ctab'] = ctab
        tvars['t_per'] = t_per
        tvars['n_obs'] = n_obs
        tvars['cmag'] = cent_mag
        tvars['tcat'] = tcat

        # Compute the measure expressing the performance of the current
        # completeness. If the norm is smaller than the previous one
        # `check` is True
        rates = [n/t for n,t in zip(n_obs, t_per)]
        stmags = [float(m) for m in cent_mag]
        check, trate, tnorm = check_criterion(criterion, rate, tnorm, tvars)
        all_res.append([iper, aval, bval, tnorm])
        all_mags.append(stmags)
        all_rates.append(rates)

        # Saving the information for the current completeness table.
        if check:
            iper_save = iper
            rate = trate
            norm = tnorm
            save = [aval, bval, rate, ctab, norm, siga, sigb,
                    min(ctab[:, 1]), rmag_rate, rmag_sigma_rate]
            gwci = get_weichert_confidence_intervals
            lcl, ucl, ex_rates, ex_rates_scaled = gwci(
                cent_mag, n_obs, t_per, bval)
            mmax = max(tcat.data['magnitude'])
            # Scheme:
            # 0, 1, 2, 3, 4
            # 5, 6, 7, 8, 9
            # 10, 11
            # 12, 13
            wei = [cent_mag, n_obs, binw, t_per, ex_rates_scaled,
                   lcl, ucl, mmax, aval, bval,
                   wei_conf['reference_magnitude'], rmag_rate,
                   rmag_sigma_rate, sigb]
            count['complete'] += 1

    # Print info
    print(f'Iteration: {iper:05d} norm: {norm:12.6e}')

    if len(save) > 0:
        print(f'Index of selected permutation : {iper_save:d}')
        print(f'Maximum annual rate for {ref_mag:.1f}   : {save[2]:.4f}')
        print(f'GR a and b                    : {save[0]:.4f} {save[1]:.4f}')
        print('Completeness:\n', save[3])
        print(count)
    else:
        print('No results')
        print(count)

    if wei is None:
        return save

    # Plotting
    _weichert_plot(wei[0], wei[1], wei[2], wei[3], wei[4], wei[5], wei[6],
                   wei[7], wei[8], wei[9], src_id=src_id, plt_show=False,
                   ref_mag=wei[10], ref_mag_rate=wei[11],
                   ref_mag_rate_sig=wei[12], bval_sigma=wei[13])

    # Saving figure
    if folder_out_figs is not None:

        if not os.path.exists(folder_out_figs):
            create_folder(folder_out_figs)
        ext = 'png'
        fmt = 'fig_mfd_{:s}.{:s}'
        figure_fname = os.path.join(folder_out_figs,
                                    fmt.format(src_id, ext))
        plt.savefig(figure_fname, format=ext)
        plt.close()

    if folder_out is not None:
        if not os.path.exists(folder_out):
            create_folder(folder_out)
        columns = ['id', 'agr', 'bgr', 'norm']
        df = pd.DataFrame(data=np.array(all_res), columns=columns)
        df['mags'] = all_mags
        df['rates'] = all_rates
        fname = os.path.join(folder_out, f'full.results_{src_id:s}.csv')
        df.to_csv(fname, index=False)

    return save


def completeness_analysis(fname_input_pattern, fname_config, folder_out_figs,
                          folder_in, folder_out, skip=''):
    """
    :param fname_input_pattern:
        Pattern to the files with the subcatalogues
    :param fname_config:
        .toml configuration file
    :param folder_out_figs:
        Output folder for figures
    :param folder_in:
        Folder with the completeness windows
    :param folder_out:
        Folder where to store results
    :param skip:
        List with the IDs of the sources to skip
    """

    # Loading configuration
    config = toml.load(fname_config)

    # Read parameters for completeness analysis
    key = 'completeness'
    mags = np.array(config[key]['mags'])
    years = np.array(config[key]['years'])
    binw = config.get('bin_width', 0.1)
    ref_mag = config[key].get('ref_mag', 5.0)
    ref_upp_mag = config[key].get('ref_upp_mag', None)
    bmin = config[key].get('bmin', 0.8)
    bmax = config[key].get('bmax', 1.2)
    # Options: 'largest_rate', 'match_rate', 'optimize'
    criterion = config[key].get('optimization_criterion', 'optimize')
    print(criterion)

    # Reading completeness data
    print(f'Reading completeness data from: {folder_in:s}')
    fname_disp = os.path.join(folder_in, 'dispositions.npy')
    perms = np.load(fname_disp)
    mags_chk = np.load(os.path.join(folder_in, 'mags.npy'))
    years_chk = np.load(os.path.join(folder_in, 'years.npy'))
    compl_tables = {'perms': perms, 'mags_chk': mags_chk,
                    'years_chk': years_chk}

    # Fixing sorting of years
    if np.all(np.diff(years)) >= 0:
        years = np.flipud(years)

    np.testing.assert_array_equal(mags, mags_chk)
    np.testing.assert_array_equal(years, years_chk)

    # Info
    if len(skip) > 0:
        if isinstance(skip, str):
            skip = get_list(skip)
        print('Skipping: ', skip)

    # Processing subcatalogues
    for fname in glob.glob(fname_input_pattern):
        # Get source ID
        src_id = _get_src_id(fname)
        # If necessary skip the source
        if src_id in skip:
            continue

        # Read configuration parameters for the current source
        if src_id in config['sources']:
            var = config['sources'][src_id]
        else:
            var = {}

        res = _completeness_analysis(fname, years, mags, binw, ref_mag,
                                     ref_upp_mag, [bmin, bmax], criterion,
                                     compl_tables, src_id,
                                     folder_out_figs=folder_out_figs,
                                     folder_out=folder_out,
                                     rewrite=False)
        #print(len(res))
        if len(res) == 0:
            continue

        # Formatting completeness table
        tmp = []
        for row in res[3]:
            tmp.append([float(row[0]), float(row[1])])
        var['completeness_table'] = tmp
        var['agr_weichert'] = float(f'{res[0]:.5f}')
        var['bgr_weichert'] = float(f'{res[1]:.5f}')
        var['agr_sig_weichert'] = float(f'{res[5]:.5f}')
        var['bgr_sig_weichert'] = float(f'{res[6]:.5f}')
        var['rmag'] = float(f'{res[7]:.5f}')
        var['rmag_rate'] = float(f'{res[8]:.5e}')
        var['rmag_rate_sig'] = float(f'{res[9]:.5e}')

        # Updating configuration
        config['sources'][src_id] = var

    with open(fname_config, 'w', encoding='utf-8') as fou:
        fou.write(toml.dumps(config))
        print(f'Updated {fname_config:s}')
