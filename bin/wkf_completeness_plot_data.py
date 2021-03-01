#!/usr/bin/env python
# coding: utf-8

import os
import toml
import numpy
import matplotlib.pyplot as plt
from openquake.baselib import sap
from glob import glob
from openquake.wkf.utils import _get_src_id, create_folder
from scipy.stats import chi2
from openquake.mbt.tools.model_building.plt_mtd import create_mtd


def get_weichert_confidence_intervals(occ, tcompl):
    exceedance_rates = numpy.array([sum(occ[i:]) for i in range(len(occ))])
    exceedance_rates_scaled = numpy.array([sum(occ[i:]/tcompl[i:]) for i in
                                           range(len(occ))])
    N = sum(occ)
    u = 0.5*chi2.ppf(0.841, 2*(N+1))
    l = 0.5*chi2.ppf(0.159, 2*N)
    return (l/max(tcompl)*tcompl, u/max(tcompl)*tcompl, exceedance_rates,
            exceedance_rates_scaled)


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
        print(fname)

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

        if ('sources' in model and
                'completeness_table' in model['sources'][src_id]):
            ctab = numpy.array(model['sources'][src_id]['completeness_table'])
        else:
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


subcatalogues_analysis.fname_input_pattern = 'Name of a shapefile with polygons'
subcatalogues_analysis.fname_config = 'Name of the .toml file with configuration parameters'
subcatalogues_analysis.outdir = 'Name of the output folder'

if __name__ == '__main__':
    sap.run(subcatalogues_analysis)
