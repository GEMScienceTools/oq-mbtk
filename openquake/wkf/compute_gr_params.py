#!/usr/bin/env python
# coding: utf-8

import os
import toml
import numpy
import pandas as pd
import geopandas as gpd
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


def compute_a_value_from_density(fname_input_pattern: str,
                                 fname_config: str, fname_polygon: str,
                                 use: str = '', folder_out_figs: str = None,
                                 plt_show=False):
    """
    This function computes the a_gr value for polygons given a rate of
    occurrence per unit of area. This rate is specified in the `fname_config`
    file.
    """

    if len(use) > 0:
        use = get_list(use)

    # Read the settings
    model = toml.load(fname_config)

    # Read the file with polygons set projection
    print(fname_polygon)
    gdf = gpd.read_file(fname_polygon)
    gdf = gdf.to_crs({'init': 'epsg:3857'})

    # Loop over polygons
    for idx, poly in gdf.iterrows():

        # Get source ID
        src_id = poly.id
        print(src_id)

        if len(use) > 0 and src_id not in use:
            print('skipping')
            continue

        # Getting area in km2
        area = poly["geometry"].area / 1e6

        # Getting rate and reference mag
        if (src_id in model['sources'] and
                'rate_basel' in model['sources'][src_id]):
            rate = float(model['sources'][src_id]['rate_basel'])
            bgr = float(model['sources'][src_id]['bgr_basel'])
            mref = float(model['sources'][src_id]['mref_basel'])
        else:
            rate = float(model['baseline']['rate_basel'])
            bgr = float(model['baseline']['bgr_basel'])
            mref = float(model['baseline']['mref_basel'])

        # Computing agr
        agr = numpy.log10(rate * area) + bgr * mref

        # Saving agr
        if 'sources' not in model:
            model['sources'] = {}
        if src_id not in model['sources']:
            model['sources'][src_id] = {}

        model['sources'][src_id]['agr_basel'] = float('{:.5f}'.format(agr))

    # Saving results into the config file
    with open(fname_config, 'w') as fou:
        fou.write(toml.dumps(model))
        print('Updated {:s}'.format(fname_config))


def get_mmax_ctab(model, src_id):

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

    return mmax, ctab


def get_exrs(df: pd.DataFrame, bgr: str):
    """
    Computes annual exceedence rates using eq. 10 in Weichert (1980; BSSA).

    :param df:
        An instance of :class:`pandas.DataFrame` with the following columns:
        - `mag`: magnitude
        - `nobs`
        - `deltaT`: duration [yr]
    :param bgr:
        The b-value of the Gutenberg-Richer relatioship
    :returns:
        Annual exceedance rate for all the magnitude values in the dataframe.
    """
    beta = bgr * numpy.log(10.0)
    exr = []
    for m in df.mag:
        cond = (df.nobs > 0) & (df.mag >= m)
        N = sum(df.nobs[cond])
        tmp = numpy.exp(-beta*df.mag[cond])
        num = numpy.sum(tmp)
        den = numpy.sum(tmp*df.deltaT[cond])
        exr.append(N * num / den)
    return numpy.array(exr)


def get_agr(mag, bgr, rate):
    """
    :param mag:
        The magnitude to which the parameter `rate` refers to. If the rates
        are binned this should be the lower limit of the bin with `mag`
    :param bgr:
        The b-value of the Gutenberg-Richer relatioship
    :param rate:
        The rate of occurrence of earthquakes larger than `mag`
    :returns:
        The a-value of the GR relationship
    """
    return numpy.log10(rate) + bgr * (mag)


def compute_a_value(fname_input_pattern: str, bval: float, fname_config: str,
                    folder_out: str, use: str = '',
                    folder_out_figs: str = None, plt_show=False):
    """
    This function assignes an a-value to each source with a file selected by
    the provided `fname_input_pattern`.

    :param fname_input_pattern:
        The name of a file or of a pattern
    :param bval:
        The b-value of the GR distribution
    :param fname_config:
        Configuration file
    :param folder_out:
        Path to the output folder
    """

    if len(use) > 0:
        use = get_list(use)

    # Processing input parameters
    bval = float(bval)
    if folder_out is not None:
        create_folder(folder_out)
    if folder_out_figs is not None:
        create_folder(folder_out_figs)

    if isinstance(fname_input_pattern, str):
        fname_list = glob(fname_input_pattern)
    else:
        fname_list = fname_input_pattern

    # Parsing config
    model = toml.load(fname_config)
    binw = model['bin_width']

    # Processing files
    for fname in sorted(fname_list):

        # Get source ID
        src_id = _get_src_id(fname)
        if len(use) > 0 and src_id not in use:
            continue
        print(fname)

        mmax, ctab = get_mmax_ctab(model, src_id)

        # Processing catalogue
        tcat = _load_catalogue(fname)

        if tcat is None or len(tcat.data['magnitude']) < 2:
            continue

        # Completeness analysis
        tcat = _add_defaults(tcat)
        tcat.data["dtime"] = tcat.get_decimal_time()
        try:
            cent_mag, t_per, n_obs = get_completeness_counts(tcat, ctab, binw)
            if cent_mag is None:
                print('   a-value calculation: completeness analysis failed')
                continue
        except ValueError:
            print('   a-value calculation: completeness analysis failed')
            continue

        df = pd.DataFrame()
        df['mag'] = cent_mag
        df['deltaT'] = t_per
        df['nobs'] = n_obs
        fout = os.path.join(folder_out, 'occ_count_zone_{:s}'.format(src_id))
        df.to_csv(fout, index=False)

        # Computing GR a
        if 'sources' not in model:
            model['sources'] = {}
        if src_id not in model['sources']:
            model['sources'][src_id] = {}

        exrs = get_exrs(df, bval)
        aval = get_agr(df.mag[0]-binw/2, bval, exrs[0])

        tmp = "{:.5e}".format(aval)
        model['sources'][src_id]['agr_counting'] = float(tmp)

        tmp = "{:.5e}".format(bval)
        model['sources'][src_id]['bgr_counting'] = float(tmp)

        gwci = get_weichert_confidence_intervals
        lcl, ucl, ex_rates, ex_rates_scaled = gwci(cent_mag, n_obs, t_per,
                                                   bval)

        _ = plt.figure()
        ax = plt.gca()
        plt.plot(cent_mag, n_obs/t_per, 'o', markerfacecolor='none')
        plt.plot(cent_mag-binw/2, ex_rates_scaled, 's', markerfacecolor='none',
                 color='red')

        plt.plot(cent_mag-binw/2, lcl, '--', color='black')
        plt.plot(cent_mag-binw/2, ucl, '--', color='black')

        xmag = numpy.arange(cent_mag[0]-binw/2, mmax-0.01*binw, binw/2)
        exra = (10.0**(aval - bval * xmag) -
                10.0**(aval - bval * mmax))
        plt.plot(xmag, exra, '--', lw=3, color='green')

        plt.yscale('log')
        plt.xlabel('Magnitude')
        plt.ylabel('Annual rate of exceedance')
        plt.text(0.75, 0.95, 'Fixed b_GR = {:.2f}'.format(bval),
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
    with open(fname_config, 'w') as fou:
        fou.write(toml.dumps(model))
        print('Updated {:s}'.format(fname_config))


def get_weichert_confidence_intervals(mag, occ, tcompl, bgr):
    """
    :param mag:
    :param occ:
    :param tcompl:
    :param bgr:
    """

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
            print('Skipping: ', skip)

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


def _weichert_analysis(tcat, ctab, binw, cent_mag, n_obs, t_per,
                       folder_out=None, src_id=None):
    """
    :param tcat:
        A catalogue instance
    :param ctab:
        Completeness table
    :param binw:
        Binw width folder
    :param:
        path to the folder where to stopre information
    :param src_id:
        The source id
    :returns:
        A tuple with a and b values,
    """

    # Computing GR a and b
    weichert_config = {'magnitude_interval': binw,
                       'reference_magnitude': 0.0}
    weichert = Weichert()
    bval_wei, sigmab, aval_wei, sigmaa = weichert.calculate(
        tcat, weichert_config, ctab)

    # Computing confidence intervals
    gwci = get_weichert_confidence_intervals
    lcl, ucl, ex_rates, ex_rates_scaled = gwci(
        cent_mag, n_obs, t_per, bval_wei)

    return aval_wei, bval_wei, lcl, ucl, ex_rates, ex_rates_scaled


def _weichert_plot(cent_mag, n_obs, binw, t_per, ex_rates_scaled,
                   lcl, ucl, mmax, aval_wei, bval_wei, src_id=None,
                   plt_show=False):

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
        print(fname, end='')

        # Get source ID
        src_id = _get_src_id(fname)
        if src_id in skip:
            print("   skipping")
            continue
        else:
            print("")

        if 'sources' in model:
            if (src_id in model['sources'] and
                    'mmax' in model['sources'][src_id]):
                mmax = model['sources'][src_id]['mmax']
            else:
                mmax = model['default']['mmax']
            if (src_id in model['sources'] and
                    'completeness_table' in model['sources'][src_id]):
                key_tmp = 'completeness_table'
                ctab = numpy.array(model['sources'][src_id][key_tmp])
                print('Using source specific completeness')
            else:
                ctab = numpy.array(model['default']['completeness_table'])
        else:
            mmax = model['default']['mmax']
            ctab = numpy.array(model['default']['completeness_table'])

        # Processing catalogue
        tcat = _load_catalogue(fname)
        if tcat is None or len(tcat.data['magnitude']) < 2:
            print('    Source {:s} has less than 2 eqks'.format(src_id))
            continue
        tcat = _add_defaults(tcat)

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

        aval, bval, lcl, ucl, ex_rates, ex_rates_scaled = _weichert_analysis(
            tcat, ctab, binw, cent_mag, n_obs, t_per, folder_out, src_id)

        _weichert_plot(cent_mag, n_obs, binw, t_per, ex_rates_scaled,
                       lcl, ucl, mmax, aval, bval, src_id, plt_show)

        if 'sources' not in model:
            model['sources'] = {}
        if src_id not in model['sources']:
            model['sources'][src_id] = {}

        tmp = "{:.5e}".format(aval)
        model['sources'][src_id]['agr_weichert'] = float(tmp)
        tmp = "{:.3f}".format(bval)
        model['sources'][src_id]['bgr_weichert'] = float(tmp)

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


def main(fname_input_pattern, fname_config, folder_out=None,
         folder_out_figs=None, *, skip=[], binw=None, plt_show=False):

    weichert_analysis(fname_input_pattern, fname_config, folder_out,
                      folder_out_figs, skip, binw, plt_show)


main.fname_input_pattern = 'Name of a shapefile with polygons'
msg = 'Name of the .toml file with configuration parameters'
main.fname_config = msg
msg = 'Name of the output folder where to store occurrence counts'
main.folder_out = msg
msg = 'Name of the output folder where to store figures'
main.folder_out_figs = msg
msg = 'A list with the ID of sources that should not be considered'
main.skip = msg
main.binw = 'Width of the magnitude bin used in the analysis'
main.plot_show = 'Show figures on screen'

if __name__ == '__main__':
    sap.run(main)
