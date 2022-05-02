#!/usr/bin/env python
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
import matplotlib.pyplot as plt

from openquake.baselib import sap
from openquake.wkf.utils import _get_src_id, create_folder
from openquake.wkf.compute_gr_params import (get_weichert_confidence_intervals,
                                             _weichert_plot)
from openquake.mbt.tools.model_building.plt_tools import _load_catalogue
from openquake.mbt.tools.model_building.dclustering import _add_defaults
from openquake.hmtk.seismicity.occurrence.utils import get_completeness_counts
from openquake.hmtk.seismicity.occurrence.weichert import Weichert

import warnings
warnings.filterwarnings("ignore")


def clean_completeness(tmp):
    ctab = []
    for m in np.unique(tmp[:, 1]):
        idx = np.nonzero(tmp[:, 1] == m)[0]
        ctab.append([tmp[max(idx), 0], tmp[max(idx), 1]])
    ctab = np.array(ctab)
    return ctab


def completeness_analysis(fname, idxs, years, mags, binw, ref_mag, bgrlim,
                          src_id, folder_out_figs, rewrite=False):

    tcat = _load_catalogue(fname)
    tcat = _add_defaults(tcat)
    tcat.data["dtime"] = tcat.get_decimal_time()
    print('\nSOURCE:', src_id)
    print('Catalogue contains {:d} events'.format(len(tcat.data['magnitude'])))

    # See http://shorturl.at/adsvA
    fname_disp = 'dispositions.npy'
    perms = np.load(fname_disp)
    mags = np.flipud(np.load('mags.npy'))
    years = np.load('years.npy')

    wei_conf = {'magnitude_interval': binw, 'reference_magnitude': 0.0}
    weichert = Weichert()
    rate = -1e10
    save = []
    mags = np.array(mags)

    for prm in perms:

        idx = prm.astype(int)
        tmp = np.array([(y, m) for y, m in zip(years, mags[idx])])
        ctab = clean_completeness(tmp)

        try:
            cent_mag, t_per, n_obs = get_completeness_counts(tcat, ctab, binw)
            bval, sigb, aval, siga = weichert.calculate(tcat, wei_conf, ctab)

            tmp_rate = 10**(-bval*ref_mag + aval)
            if tmp_rate > rate and bval <= bgrlim[1] and bval >= bgrlim[0]:
                rate = tmp_rate
                save = [aval, bval, rate, ctab]

                gwci = get_weichert_confidence_intervals
                lcl, ucl, ex_rates, ex_rates_scaled = gwci(
                    cent_mag, n_obs, t_per, bval)

                mmax = max(tcat.data['magnitude'])
                wei = [cent_mag, n_obs, binw, t_per, ex_rates_scaled,
                       lcl, ucl, mmax, aval, bval]

        except RuntimeWarning:
            logging.debug('Skipping', ctab)

        except UserWarning:
            logging.debug('Skipping', ctab)

        except:
            logging.debug('Skipping', ctab)

    if True:
        fmt = 'Maximum annual rate for {:.1f}: {:.4f}'
        print(fmt.format(ref_mag, save[2]))
        fmt = 'GR a and b                 : {:.4f} {:.4f}'
        print(fmt.format(save[0], save[1]))
        print('Completeness:\n', save[3])

    _weichert_plot(wei[0], wei[1], wei[2], wei[3], wei[4], wei[5], wei[6],
                   wei[7], wei[8], wei[9], src_id=src_id)

    # Saving figure
    if folder_out_figs is not None:

        create_folder(folder_out_figs)

        ext = 'png'
        fmt = 'fig_mfd_{:s}.{:s}'
        figure_fname = os.path.join(folder_out_figs,
                                    fmt.format(src_id, ext))
        plt.savefig(figure_fname, format=ext)
        plt.close()

    return save


def main(fname_input_pattern, fname_config, folder_out_figs):
    """
    Analyzes the completeness of a catalogue
    """

    config = toml.load(fname_config)
    tmp = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990]
    years = np.array(tmp)
    mags = np.array([4.0, 4.5, 5, 5.5, 6, 6.5, 7])
    binw = 0.2
    ref_mag = 5.0

    # Mags in descending order
    years[::-1].sort()
    idxs = np.arange(len(mags))
    idxs[::-1].sort()
    bmin = 0.85
    bmax = 1.15
    bmin = 0.90
    bmax = 1.10

    for fname in glob.glob(fname_input_pattern):
        src_id = _get_src_id(fname)
        var = config['sources'][src_id]
        res = completeness_analysis(fname, idxs, years, mags, binw, ref_mag,
                                    [bmin, bmax], src_id, folder_out_figs,
                                    rewrite=False)
        var['completeness_table'] = list(res[3])
        var['agr_weichert'] = float('{:.4f}'.format(res[0]))
        var['bgr_weichert'] = float('{:.4f}'.format(res[1]))

    with open(fname_config, 'w') as fou:
        fou.write(toml.dumps(config))
        print('Updated {:s}'.format(fname_config))


descr = 'Pattern to select input files with subcatalogues'
main.fname_input_pattern = descr
msg = 'Name of the .toml file with configuration parameters'
main.fname_config = msg
msg = 'Name of the folder where to store figures'
main.fname_config = msg


if __name__ == '__main__':
    sap.run(main)
