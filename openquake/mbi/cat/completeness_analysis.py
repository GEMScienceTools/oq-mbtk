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

from openquake.baselib import sap
from openquake.wkf.utils import get_list
from openquake.wkf.utils import _get_src_id

from openquake.cat.completeness.analysis import completeness_analysis


def _main(fname_input_pattern, fname_config, folder_out_figs, folder_in,
          folder_out, skip=''):

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

    # Reading completeness data
    print('Reading completeness data from: {:s}'.format(folder_in))
    fname_disp = os.path.join(folder_in, 'dispositions.npy')
    perms = np.load(fname_disp)
    mags_chk = np.load(os.path.join(folder_in, 'mags.npy'))
    years_chk = np.load(os.path.join(folder_in, 'years.npy'))
    compl_tables = {'perms': perms, 'mags_chk': mags_chk,
                    'years_chk': years_chk}

    # Fixing sorting of years
    if not np.all(np.diff(years)) < 0:
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

        res = completeness_analysis(fname, years, mags, binw, ref_mag,
                                    ref_upp_mag, [bmin, bmax], criterion,
                                    compl_tables, src_id, folder_out,
                                    rewrite=False)

        # Formatting completeness table
        tmp = []
        for row in res[3]:
            tmp.append([float(row[0]), float(row[1])])
        var['completeness_table'] = tmp
        var['agr_weichert'] = float('{:.4f}'.format(res[0]))
        var['bgr_weichert'] = float('{:.4f}'.format(res[1]))

        # Updating configuration
        config['sources'][src_id] = var

    with open(fname_config, 'w') as fou:
        fou.write(toml.dumps(config))
        print('Updated {:s}'.format(fname_config))


def main(fname_input_pattern, fname_config, folder_out_figs, folder_in,
         folder_out, *, skip: str = ''):
    """
    Analyzes the completeness of a catalogue
    """

    _main(fname_input_pattern, fname_config, folder_out_figs, folder_in,
          folder_out, skip)


descr = 'Pattern to select input files with subcatalogues'
main.fname_input_pattern = descr
msg = 'Name of the .toml file with configuration parameters'
main.fname_config = msg
msg = 'Name of the folder where to store figures'
main.folder_out_figs = msg
msg = 'Name of the folder where to read candidate completeness tables'
main.folder_in = msg
msg = 'Name of the folder where to store data'
main.folder_out = msg
msg = 'IDs of the sources to skip'
main.skip = msg


if __name__ == '__main__':
    sap.run(main)
