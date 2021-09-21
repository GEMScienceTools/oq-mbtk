#!/usr/bin/env python
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
    Analyze the catalogue
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
            tmp = tmp.astype(numpy.float)
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
            ctab = ctab.astype(numpy.float)
        else:
            print(' default completeness')
            ctab = numpy.array(model['default']['completeness_table'])
            ctab = ctab.astype(numpy.float)

        print(ctab)
        _plot_ctab(ctab)

        ext = 'png'
        figure_fname = os.path.join(outdir,
                                    'fig_mtd_{:s}.{:s}'.format(src_id, ext))
        plt.savefig(figure_fname, format=ext)
        plt.close()
