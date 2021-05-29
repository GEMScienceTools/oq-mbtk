#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob

import toml
import numpy
import matplotlib.pyplot as plt
from openquake.baselib import sap
from openquake.wkf.utils import _get_src_id, create_folder
from openquake.mbt.tools.model_building.plt_mtd import create_mtd


def subcatalogues_analysis(fname_input_pattern, fname_config, outdir, skip=[],
                           **kwargs):
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
        out = create_mtd(fname, src_id, None, False, False, 0.5, 10,
                         pmint=1900)

        if out is None:
            continue

        if 'xlim' in kwargs:
            plt.xlim(kwargs['xlim'])

        if 'ylim' in kwargs:
            plt.ylim(kwargs['ylim'])

        print('src_id: {:s} '.format(src_id), end='')
        if ('sources' in model and src_id in model['sources'] and
                'completeness_table' in model['sources'][src_id]):
            print(' source specific completeness')
            ctab = numpy.array(model['sources'][src_id]['completeness_table'])
        else:
            print(' default completeness')
            ctab = numpy.array(model['default']['completeness_table'])

        n = len(ctab)
        for i in range(0, n-1):
            plt.plot([ctab[i, 0], ctab[i, 0]], [ctab[i, 1],
                     ctab[i+1, 1]], '-r')
            plt.plot([ctab[i, 0], ctab[i+1, 0]], [ctab[i+1, 1],
                     ctab[i+1, 1]], '-r')

        ylim = plt.gca().get_ylim()
        xlim = plt.gca().get_xlim()

        plt.plot([ctab[n-1, 0], ctab[n-1, 0]], [ylim[1], ctab[n-1, 1]], '-r')
        plt.plot([ctab[0, 0], xlim[1]], [ctab[0, 1], ctab[0, 1]], '-r')

        ext = 'png'
        figure_fname = os.path.join(outdir,
                                    'fig_mtd_{:s}.{:s}'.format(src_id, ext))
        plt.savefig(figure_fname, format=ext)
        plt.close()


descr = 'Name of a shapefile with polygons'
subcatalogues_analysis.fname_input_pattern = descr
descr = 'Name of the .toml file with configuration parameters'
subcatalogues_analysis.fname_config = descr
subcatalogues_analysis.outdir = 'Name of the output folder'

if __name__ == '__main__':
    sap.run(subcatalogues_analysis)
