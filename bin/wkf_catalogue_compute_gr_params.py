#!/usr/bin/env python
# coding: utf-8

import os
import toml
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from openquake.wkf.utils import _get_src_id, create_folder, get_list
from scipy.stats import chi2
from openquake.baselib import sap
from openquake.mbt.tools.model_building.dclustering import _add_defaults
from openquake.mbt.tools.model_building.plt_tools import _load_catalogue
from openquake.mbt.tools.model_building.plt_mtd import create_mtd
from openquake.hmtk.seismicity.occurrence.utils import get_completeness_counts
from openquake.hmtk.seismicity.occurrence.weichert import Weichert


def get_weichert_confidence_intervals(mag, occ, tcompl, bgr):

    beta = bgr * numpy.log(10.0)
    N = numpy.array([sum(occ[i:]) for i in range(len(occ))])

    exceedance_rates_scaled = numpy.zeros_like(mag)
    lob = numpy.zeros_like(mag)
    upb = numpy.zeros_like(mag)
    for i in range(len(occ)):
        num = numpy.sum(numpy.exp(-beta*mag[i:]))
        den = numpy.sum(tcompl[i:]*numpy.exp(-beta*mag[i:]))
        exceedance_rates_scaled[i] = N[i] * num / den
        lob[i] = 0.5*chi2.ppf(0.841, 2*(N[i]+1)) * num / den
        upb[i] = 0.5*chi2.ppf(0.159, 2*N[i]) * num / den

    return (lob, upb, N, exceedance_rates_scaled)


def subcatalogues_analysis(fname_input_pattern, fname_config, skip=[],
                           outdir='', **kwargs):
    """
    Analyze the catalogues included in a folder.
    """

    if len(skip) > 0:
        if isinstance(skip, str):
            skip = get_list(skip)
        print(skip)

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

        ylim = plt.gca().get_ylim()
        xlim = plt.gca().get_xlim()

        if ('sources' in model and
                'completeness_table' in model['sources'][src_id]):
            ctab = numpy.array(model['sources'][src_id]['completeness_table'])
            print('Using source specific completeness')
        else:
            ctab = numpy.array(model['default']['completeness_table'])

        n = len(ctab)
        for i in range(0, n-1):
            plt.plot([ctab[i, 0], ctab[i, 0]], [ctab[i, 1],
                     ctab[i+1, 1]], '-r')
            plt.plot([ctab[i, 0], ctab[i+1, 0]], [ctab[i+1, 1],
                     ctab[i+1, 1]], '-r')
        plt.plot([ctab[n-1, 0], ctab[n-1, 0]], [ylim[1], ctab[n-1, 1]], '-r')
        plt.plot([ctab[0, 0], xlim[1]], [ctab[0, 1], ctab[0, 1]], '-r')

        ext = 'png'
        figure_fname = os.path.join(outdir,
                                    'fig_mtd_{:s}.{:s}'.format(src_id, ext))
        plt.savefig(figure_fname, format=ext)
        plt.close()

        break


def weichert_analysis(fname_input_pattern, fname_config, folder_out=None,
                      folder_out_figs=None, skip=[], binw=None,
                      plt_show=False):
    """
    Computes GR parameters for a set of catalogues stored in a .csv file

    :param fname_input_pattern:
        It can be either a string (definining a pattern) or a list of
        .csv files. The file names must have the source ID at the end. The
        delimiter of the source ID on the left is `_`
    :param fname_config:
        The name of the .toml configuration file
    :param folder_out:
        The folder where to store the files with the counting of occurrences
    :param folder_out_figs:
        The folder where to store the figures
    :param skip:
        A list with the IDs of the sources to skip
    """

    if folder_out is not None:
        create_folder(folder_out)
    if folder_out_figs is not None:
        create_folder(folder_out_figs)

    # Parsing config
    if fname_config is not None:
        model = toml.load(fname_config)

    if binw is None and fname_config is not None:
        binw = model['bin_width']
    else:
        binw = 0.1

    if isinstance(fname_input_pattern, str):
        fname_list = [f for f in glob(fname_input_pattern)]
    else:
        fname_list = fname_input_pattern

    # Processing files
    for fname in sorted(fname_list):

        # Get source ID
        src_id = _get_src_id(fname)
        if src_id in skip:
            continue
        print(fname)

        if 'sources' in model:
            if (src_id in model['sources'] and
                    'mmax' in model['sources'][src_id]):
                mmax = model['sources'][src_id]['mmax']
            else:
                mmax = model['default']['mmax']
            if (src_id in model['sources'] and
                    'completeness_table' in model['sources'][src_id]):
                ctab = numpy.array(model['sources'][src_id]['completeness_table'])
                print('Using source specific completeness')
            else:
                ctab = numpy.array(model['default']['completeness_table'])
        else:
            mmax = model['default']['mmax']
            ctab = numpy.array(model['default']['completeness_table'])

        # Processing catalogue
        tcat = _load_catalogue(fname)

        if tcat is None or len(tcat.data['magnitude']) < 2:
            return None

        tcat.data["dtime"] = tcat.get_decimal_time()
        cent_mag, t_per, n_obs = get_completeness_counts(tcat, ctab, binw)

        if folder_out is not None:
            df = pd.DataFrame()
            df['mag'] = cent_mag
            df['deltaT'] = t_per
            df['nobs'] = n_obs
            fmt = 'occ_count_zone_{:s}'
            fout = os.path.join(folder_out, fmt.format(src_id))
            df.to_csv(fout, index=False)

        # Computing GR a and b
        tcat = _add_defaults(tcat)
        weichert_config = {'magnitude_interval': binw,
                           'reference_magnitude': 0.0}
        weichert = Weichert()
        bval_wei, sigmab, aval_wei, sigmaa = weichert.calculate(tcat,
                weichert_config, ctab)

        # Computing confidence intervals
        gwci = get_weichert_confidence_intervals
        lcl, ucl, ex_rates, ex_rates_scaled = gwci(cent_mag, n_obs, t_per, bval_wei)

        if 'sources' not in model:
            model['sources'] = {}
        if src_id not in model['sources']:
            model['sources'][src_id] = {}

        tmp = "{:.5e}".format(aval_wei)
        model['sources'][src_id]['agr_weichert'] = float(tmp)
        tmp = "{:.3f}".format(bval_wei)
        model['sources'][src_id]['bgr_weichert'] = float(tmp)

        _ = plt.figure()
        ax = plt.gca()
        plt.plot(cent_mag, n_obs/t_per, 'o', markerfacecolor='none')
        plt.plot(cent_mag-binw/2, ex_rates_scaled, 's', markerfacecolor='none',
                 color='red')

        plt.plot(cent_mag-binw/2, lcl, '--', color='darkgrey')
        plt.plot(cent_mag-binw/2, ucl, '--', color='darkgrey')

        xmag = numpy.arange(cent_mag[0]-binw/2, mmax-0.01*binw, binw/2)
        exra = (10.0**(aval_wei - bval_wei * xmag) -
                10.0**(aval_wei - bval_wei * mmax))
        plt.plot(xmag, exra, '--', lw=3, color='green')

        plt.yscale('log')
        plt.xlabel('Magnitude')
        plt.ylabel('Annual rate of exceedance')
        plt.text(0.75, 0.95, 'b_GR = {:.2f}'.format(bval_wei),
                 transform=ax.transAxes)
        plt.grid(which='major', color='grey')
        plt.grid(which='minor', linestyle='--', color='lightgrey')
        plt.title(src_id)

        if plt_show:
            plt.show()

        # Saving figures
        if folder_out_figs is not None:
            ext = 'png'
            fmt = 'fig_mfd_{:s}.{:s}'
            figure_fname = os.path.join(folder_out_figs,
                                        fmt.format(src_id, ext))

            plt.savefig(figure_fname, format=ext)
            plt.close()

    # Saving results into the config file
    if fname_config is not None:
        with open(fname_config, 'w') as f:
            f.write(toml.dumps(model))
            print('Updated {:s}'.format(fname_config))


weichert_analysis.fname_input_pattern = 'Name of a shapefile with polygons'
msg = 'Name of the .toml file with configuration parameters'
weichert_analysis.fname_config = msg
msg = 'Name of the output folder where to store occurrence counts'
weichert_analysis.folder_out = msg
msg = 'Name of the output folder where to store figures'
weichert_analysis.folder_out_figs = msg
msg = 'A list with the ID of sources that should not be considered'
weichert_analysis.skip = msg
weichert_analysis.binw = 'Width of the magnitude bin used in the analysis'
weichert_analysis.plot_show = 'Show figures on screen'

if __name__ == '__main__':
    sap.run(weichert_analysis)
