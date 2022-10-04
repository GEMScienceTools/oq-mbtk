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
import toml
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt

from openquake.cat.completeness.norms import (
    get_norm_optimize, get_norm_optimize_a, get_norm_optimize_b)
from openquake.wkf.utils import _get_src_id, create_folder, get_list
from openquake.baselib import sap
from openquake.wkf.compute_gr_params import (get_weichert_confidence_intervals,
                                             _weichert_plot)
from openquake.mbt.tools.model_building.plt_tools import _load_catalogue
from openquake.mbt.tools.model_building.dclustering import _add_defaults
from openquake.hmtk.seismicity.occurrence.utils import get_completeness_counts
from openquake.hmtk.seismicity.occurrence.weichert import Weichert

import warnings
warnings.filterwarnings("ignore")


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
    mags = np.array(list(ctab[:, 1])+[10])
    idx = np.digitize(c_mag, bins=mags, right=False)

    # low_yea = np.ones((ctab.shape[0])) * np.NaN
    low_yea = []
    for i in range(ctab.shape[0]):
        years = c_yea[idx == i]
        if len(years) >= occ_threshold:
            low_yea.append(np.sort(years)[occ_threshold-1])
        else:
            low_yea.append(np.NaN)
    return np.array(low_yea)


def clean_completeness(tmp):
    """
    The completeness table that must be simplified

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
    t_per = tvars['t_per']
    n_obs = tvars['n_obs']
    cmag = tvars['cmag']
    tcat = tvars['tcat']
    last_year = tvars['last_year']

    if criterion == 'largest_rate':

        # Computes the rate to be maximised
        tmp_rate = 10**(-bval * ref_mag + aval)
        if ref_upp_mag is not None:
            tmp_rate -= 10**(-bval * ref_upp_mag + aval)
        norm = 1./abs(tmp_rate)

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
        # norm = get_norm_optimize(aval, bval, ctab, cmag, t_per, n_obs,
        #                         last_year, info=False)
        # norm = get_norm_optimize_a(aval, bval, ctab, cmag, t_per, n_obs,
        #                           binw, info=False)
        norm = get_norm_optimize_b(aval, bval, ctab, tcat, binw, ybinw=10.,
                                   mmin=ref_mag, mmax=ref_upp_mag)

    if norm is None:
        return False, -1, previous_norm

    if previous_norm > norm and bval <= bgrlim[1] and bval >= bgrlim[0]:
        check = True

    return check, tmp_rate, norm


def completeness_analysis(fname, years, mags, binw, ref_mag, ref_upp_mag,
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
    if criterion not in ['match_rate', 'largest_rate', 'optimize']:
        raise ValueError('Unknown optimization criterion')

    tcat = _load_catalogue(fname)
    tcat = _add_defaults(tcat)
    tcat.data["dtime"] = tcat.get_decimal_time()

    # Info
    mag_low = 5.0
    idx = tcat.data["magnitude"] >= mag_low
    fmt = 'Catalogue contains {:d} events equal or above {:.1f}'
    print('\nSOURCE:', src_id)
    print(fmt.format(sum(idx), mag_low))

    # Loading all the completeness tables to be considered in the analysis
    # See http://shorturl.at/adsvA
    perms = compl_tables['perms']

    # Configuration parameters for the Weichert method
    wei_conf = {'magnitude_interval': binw, 'reference_magnitude': None,
                'bvalue': 1.0}
    weichert = Weichert()

    # Initial settings
    rate = -1e10
    norm = 1e100
    save = []
    count = {'complete': 0, 'warning': 0, 'else': 0}

    all_res = []
    for iper, prm in enumerate(perms):

        print('Iteration: {:05d} norm: {:12.6e}'.format(iper, norm), end="\r")

        tmp = []
        for y, j in zip(years, prm):
            if j >= -1e-10:
                tmp.append([y, mags[int(j)]])
        tmp = np.array(tmp)
        ctab = clean_completeness(tmp)

        # Check compatibility between catalogue and completeness table. This
        # function finds in each magnitude interval defined in the completeness
        # table the earliest year since the occurreence of a number X of
        # earthquakes. This ensures that the completeness table applies only to
        # sets with a number of occurrences sufficient to infer a recurrence
        # interval.
        earliest_yea = get_earliest_year_with_n_occurrences(ctab, tcat, 2)

        # Select the completeness windows using the criteria just defined
        if np.any(np.isnan(earliest_yea)) or np.any(ctab[:, 0] < earliest_yea):
            count['else'] += 1
            logging.debug('Skipping', ctab)
            continue

        # Check that the selected completeness window has decreasing years and
        # increasing magnitudes
        assert np.all(np.diff(ctab[:, 0]) <= 0)
        assert np.all(np.diff(ctab[:, 1]) >= 0)

        # Compute occurrence
        if True:

            cent_mag, t_per, n_obs = get_completeness_counts(tcat, ctab, binw)
            bval, sigb, aval, siga = weichert.calculate(tcat, wei_conf, ctab)

            if bval >= bgrlim[1] or bval <= bgrlim[0]:
                count['else'] += 1
                continue

            r_mag = np.floor((ref_mag+binw*0.01)/binw)*binw-binw/2
            r_upp_mag = np.ceil((ref_upp_mag+binw*0.01)/binw)*binw+binw/2

            # Create a dictionary of parameters for the function which computed
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
            tvars['mmax_within_range'] = np.max(cent_mag[idx_obs])
            tvars['ctab'] = ctab
            tvars['t_per'] = t_per
            tvars['n_obs'] = n_obs
            tvars['cmag'] = cent_mag
            tvars['tcat'] = tcat

            # Compute the measure expressing the performance of the current
            # completeness. If the norm is smaller than the previous one we
            # save the information.
            check, trate, tnorm = check_criterion(
                criterion, rate, norm, tvars)
            all_res.append([iper, aval, bval, tnorm])

            # Saving the information for the current completeness table.
            if check:
                rate = trate
                norm = tnorm
                save = [aval, bval, rate, ctab, norm, siga, sigb]
                gwci = get_weichert_confidence_intervals
                lcl, ucl, ex_rates, ex_rates_scaled = gwci(
                    cent_mag, n_obs, t_per, bval)
                mmax = max(tcat.data['magnitude'])
                wei = [cent_mag, n_obs, binw, t_per, ex_rates_scaled,
                       lcl, ucl, mmax, aval, bval]

        try:
            count['complete'] += 1

        except RuntimeWarning:
            count['warning'] += 1
            logging.debug('Skipping', ctab)

        except UserWarning:
            count['warning'] += 1
            logging.debug('Skipping', ctab)

        except:
            count['else'] += 1
            logging.debug('Skipping', ctab)

    # Print info
    print('Iteration: {:05d} norm: {:12.6e}'.format(iper, norm))

    if True and len(save):
        fmt = 'Maximum annual rate for {:.1f}: {:.4f}'
        print(fmt.format(ref_mag, save[2]))
        fmt = 'GR a and b                 : {:.4f} {:.4f}'
        print(fmt.format(save[0], save[1]))
        print('Completeness:\n', save[3])
        print(count)
    else:
        print('No results')
        print(count)

    _weichert_plot(wei[0], wei[1], wei[2], wei[3], wei[4], wei[5], wei[6],
                   wei[7], wei[8], wei[9], src_id=src_id)

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
        fname = os.path.join(folder_out, f'full.results_{src_id:s}.csv')
        df.to_csv(fname, index=False)

    return save


def main(fname_input_pattern, fname_config, folder_out, *, skip=[],
         folder_out_data=None, in_folder='.'):

    if folder_out_data is not None:
        create_folder(folder_out_data)

    if len(skip) > 0:
        if isinstance(skip, str):
            skip = get_list(skip)
            print('Skipping: ', skip)

    config = toml.load(fname_config)

    key = 'completeness'
    mags = np.array(config[key]['mags'])
    years = np.array(config[key]['years'])

    binw = config.get('bin_width', 0.1)
    ref_mag = config[key].get('ref_mag', 5.0)
    ref_upp_mag = config[key].get('ref_upp_mag', None)
    bmin = config[key].get('bmin', 0.8)
    bmax = config[key].get('bmax', 1.2)
    criterion = config[key].get('optimization_criterion', 'largest_rate')

    # Mags in descending order
    years[::-1].sort()

    if 'sources' not in config:
        config['sources'] = {}

    print('Reading completeness data from: {:s}'.format(in_folder))
    fname_disp = os.path.join(in_folder, 'dispositions.npy')
    perms = np.load(fname_disp)
    mags_chk = np.load(os.path.join(in_folder, 'mags.npy'))
    years_chk = np.load(os.path.join(in_folder, 'years.npy'))
    compl_tables = {'perms': perms, 'mags_chk': mags_chk,
                    'years_chk': years_chk}
    np.testing.assert_array_equal(mags, mags_chk)
    np.testing.assert_array_equal(years, years_chk)

    for fname in sorted(glob.glob(fname_input_pattern)):

        ref_mag = config[key].get('ref_mag', 5.0)
        ref_upp_mag = config[key].get('ref_upp_mag', None)

        # Get source ID
        src_id = _get_src_id(fname)
        if src_id in skip:
            continue

        src_id = _get_src_id(fname)
        if src_id not in config['sources']:
            config['sources'][src_id] = {}

        var = config['sources'][src_id]
        res, ares = completeness_analysis(fname, years, mags, binw, ref_mag,
                                          ref_upp_mag, [bmin, bmax], criterion,
                                          compl_tables, src_id, folder_out,
                                          rewrite=False)

        var['completeness_table'] = list(res[3])
        var['agr_weichert'] = float('{:.4f}'.format(res[0]))
        var['bgr_weichert'] = float('{:.4f}'.format(res[1]))

        if folder_out_data is not None:
            tmpname = os.path.join(folder_out_data, 'data_{:s}'.format(src_id))
            np.save(tmpname, ares)

    with open(fname_config, 'w') as fou:
        fou.write(toml.dumps(config))
        print('Updated {:s}'.format(fname_config))


descr = 'Pattern to select input files with subcatalogues'
main.fname_input_pattern = descr
msg = 'Name of the .toml file with configuration parameters'
main.fname_config = msg
msg = 'Name of the folder where to store figures'
main.folder_out = msg
msg = 'A list with the ID of sources that should not be considered'
main.skip = msg
msg = 'Name of the folder where to store data'
main.folder_out_data = msg
msg = 'Name of the folder where to read .npy files with completeness tables'
main.in_folder = msg

if __name__ == '__main__':
    sap.run(main)
