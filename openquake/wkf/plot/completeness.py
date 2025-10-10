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
from glob import glob

import toml
import numpy
import matplotlib.pyplot as plt
from openquake.baselib import sap
from openquake.wkf.utils import _get_src_id, create_folder
from openquake.wkf.completeness import _plot_ctab
from openquake.mbt.tools.model_building.plt_mtd import create_mtd


def completeness_plot(fname_input_pattern, fname_config, outdir, skip=[],
                      yealim='', **kwargs):
    """
    Plot the density of events in time using `mbt.tools.model_building.plot_mtd`.
    
    :param fname_input: 
    	name of input file for catalogue
    :param fname_config:
    	configuration file, should contain a `completeness_table` for each zone to be plotted
    	or a default completeness_table to be used
    :param outdir:
    	Output folder for figures
    :param skip:
    	optional list of source IDs to skip
    :param yealim:
    	optional upper and lower year limits
    """

    create_folder(outdir)

    # Parsing config
    model = toml.load(fname_config)

    # Processing files
    for fname in sorted(glob(fname_input_pattern)):

        # Get source ID
        src_id = _get_src_id(fname)
        if src_id in skip:
            continue

        # Create figure
        out = create_mtd(fname, src_id, None, False, False, 0.25, 10,
                         pmint=1900)

        if out is None:
            continue

        if len(yealim) > 0:
            tmp = yealim.split(',')
            tmp = numpy.array(tmp)
            tmp = tmp.astype(float)
            plt.xlim(tmp)

        if 'xlim' in kwargs:
            plt.xlim(kwargs['xlim'])

        if 'ylim' in kwargs:
            plt.ylim(kwargs['ylim'])

        print('src_id: {:s} '.format(src_id), end='')
        if ('sources' in model and src_id in model['sources'] and
                'completeness_table' in model['sources'][src_id]):
            print(' source specific completeness')
            ctab = numpy.array(model['sources'][src_id]['completeness_table'])
            ctab = ctab.astype(float)
        else:
            print(' default completeness')
            ctab = numpy.array(model['default']['completeness_table'])
            ctab = ctab.astype(float)

        print(ctab)
        _plot_ctab(ctab)

        ext = 'png'
        figure_fname = os.path.join(outdir,
                                    'fig_mtd_{:s}.{:s}'.format(src_id, ext))
        plt.savefig(figure_fname, format=ext)
        plt.close()
