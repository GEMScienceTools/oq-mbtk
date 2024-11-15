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
import re
import toml
import numpy
import pathlib
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from glob import glob
from openquake.wkf.utils import _get_src_id, create_folder, get_list
from scipy.stats import chi2
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


def get_mmax_ctab(model, src_id):

    if 'sources' in model:
        if (src_id in model['sources'] and
                'mmax' in model['sources'][src_id]):
            mmax = model['sources'][src_id]['mmax']
        else:
            print(f'{src_id} misses mmax')
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
        are binned this should be the lower limit of the bin containing `mag`
    :param bgr:
        The b-value of the Gutenberg-Richer relatioship
    :param rate:
        The rate of occurrence of earthquakes larger than `mag`
    :returns:
        The a-value of the GR relationship
    """
    return numpy.log10(rate) + bgr * (mag)


def _compute_a_value(tcat, ctab, bval, binw):

    if tcat is None or len(tcat.data['magnitude']) < 2:
        return None, None, None, None

    # Completeness analysis
    tcat = _add_defaults(tcat)
    tcat.data["dtime"] = tcat.get_decimal_time()
    try:
        cent_mag, t_per, n_obs = get_completeness_counts(tcat, ctab, binw)
        if cent_mag is None:
            print('   a-value calculation: completeness analysis failed')
            return None, None, None, None
    except ValueError:
        print('   a-value calculation: completeness analysis failed')
        return None, None, None, None

    df = pd.DataFrame()
    df['mag'] = cent_mag
    df['deltaT'] = t_per
    df['nobs'] = n_obs

    # Computing GR a. 'exrs' corresponds to N_alpha
    exrs = get_exrs(df, bval)
    aval = get_agr(df.mag[0]-binw/2, bval, exrs[0])

    return aval, cent_mag, n_obs, t_per, df


def compute_a_value(fname_input_pattern: str, bval: float, fname_config: str,
                    folder_out: str, use: str = '',
                    folder_out_figs: str = None, plt_show=False,
                    src_id_pattern: str = None):
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
        if src_id_pattern is not None:
            tpath = pathlib.Path(fname)
            mtch = re.match(src_id_pattern, tpath.stem)
            src_id = mtch.group(1)
        else:
            src_id = _get_src_id(fname)

        if len(use) > 0 and src_id not in use:
            continue

        # Processing catalogue
        tcat = _load_catalogue(fname)
        mmax, ctab = get_mmax_ctab(model, src_id)
        aval, cmag, n_obs, t_per, df = _compute_a_value(tcat, ctab, bval, binw)

        if 'sources' not in model:
            model['sources'] = {}
        if src_id not in model['sources']:
            model['sources'][src_id] = {}

        tmp = "{:.5e}".format(aval)
        model['sources'][src_id]['agr_counting'] = float(tmp)
        tmp = "{:.5e}".format(bval)
        model['sources'][src_id]['bgr_counting'] = float(tmp)

        # Computing confidence intervals
        gwci = get_weichert_confidence_intervals
        lcl, ucl, ex_rates, ex_rates_scaled = gwci(
            cmag, n_obs, t_per, bval)

        # Saving results
        fout = os.path.join(folder_out, 'occ_count_zone_{:s}'.format(src_id))
        df.to_csv(fout, index=False)

        # Plotting
        _ = plt.figure()
        ax = plt.gca()
        plt.plot(cmag, n_obs/t_per, 'o', markerfacecolor='none',
                 label='Incremental rates')
        plt.plot(cmag-binw/2, ex_rates_scaled, 's', markerfacecolor='none',
                 color='red', label='Cumulative rates')

        plt.plot(cmag-binw/2, lcl, '--', color='grey', label='16th C.I.')
        plt.plot(cmag-binw/2, ucl, '-.', color='grey', label='84th C.I.')

        xmag = numpy.arange(cmag[0]-binw/2, mmax-0.01*binw, binw/2)
        exra = (10.0**(aval - bval * xmag) -
                10.0**(aval - bval * mmax))
        plt.plot(xmag, exra, '--', lw=3, color='green')

        plt.yscale('log')
        plt.xlabel('Magnitude')
        plt.ylabel('Annual rate of exceedance')
        plt.text(0.70, 0.95, 'b_GR = {:.2f} (fixed)'.format(bval),
                 transform=ax.transAxes)
        plt.text(0.70, 0.90, 'a_GR = {:.2f}'.format(aval),
                 transform=ax.transAxes)
        plt.grid(which='major', color='grey')
        plt.grid(which='minor', linestyle='--', color='lightgrey')
        plt.title(src_id)
        plt.legend(fontsize=10, loc=3)

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
    with open(fname_config, 'w', encoding='utf-8') as fou:
        fou.write(toml.dumps(model))


def get_weichert_confidence_intervals(mag, occ, tcompl, bgr):
    """
    Computes 16th-84th confidence intervals according to Weichert (1980)

    :param mag:
        A vector with the magnitude value of each magnitude bin
    :param occ:
        The number of occurrences in each magnitude bin
    :param tcompl:
        Duration [in years] of the completeness interval for each magnitude
        bin
    :param bgr:
        The GR b-value
    :returns:
        A tuple with: the upper and lower confidence interval, a vector with
        the occurrence rates and the occurrence rates scaled
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


def _weichert_analysis(tcat, ctab, binw, cmag, n_obs, t_per):
    """
    :param tcat:
        A catalogue instance
    :param ctab:
        Completeness table
    :param binw:
        Binw width
    :param cent_mag:
        path to the folder where to stopre information
    :param n_obs:
        number of observations per bin
    :param t_per:
        Duration of completeness interval per each bin
    :returns:
        A tuple with a and b values, upper and lower limits of the 16th-84th
        confidence interval, exceedance rates and exceedance rates scaled
    """

    # Computing GR a and b
    weichert_config = {'magnitude_interval': binw,
                       'reference_magnitude': numpy.min(ctab[:, 1])}
    weichert = Weichert()
    
    nev = len(tcat.data['magnitude'])
    if nev < 10:
        print("Few events in this catalogue (only ", nev, " events above completeness)") 

    # weichert.calculate returns bGR and its standard deviation + log10(rate)
    # for the reference magnitude and its standard deviation. In this case
    # we set the reference magnitude to 0 hence we get the aGR.
    fun = weichert._calculate
    # bval, sigmab, aval, sigmaa = fun(tcat, weichert_config, ctab)
    bval, sigmab, rmag_rate, rmag_rate_sigma, aval, sigmaa = fun(
        tcat, weichert_config, ctab)
        
    if bval < 0.5 or bval > 2:
        print("suspicious b-value, recheck your catalogue (b = ", bval, ")")

    # Computing confidence intervals
    gwci = get_weichert_confidence_intervals
    lcl, ucl, exrates, exrates_scaled = gwci(cmag-binw/2, n_obs, t_per, bval)

    rmag = weichert_config['reference_magnitude']
    return (aval, bval, lcl, ucl, exrates, exrates_scaled, rmag, rmag_rate,
            rmag_rate_sigma, sigmab, sigmaa)


def _get_gr_double_trunc_exceedance_rates(agr, bgr, cmag, binw, mmax):
    """
    Computes exceedance rates for a double truncated GR

    :param agr:
        GR a-value
    :param bgr:
        GR b-value
    :param cmag:
        List or array with the values of magnitude at the center of each bin
    :param binw:
        The width of the bins used to discretize magnitude
    :param mmax:
        The maximum value of magnitude
    :returns:
        The magnitude values and the corresponding exceedance rates
    """
    xmag = numpy.arange(cmag[0]-binw/2, mmax-0.01*binw, binw/2)
    exra = (10.0**(agr - bgr * xmag) -
            10.0**(agr - bgr * mmax))
    return xmag, exra


def _get_agr(bgr, rate, mag, mmax=None):
    """ Get the agr given the rate of exceedance at a given magnitude """
    den = 10**(-bgr*mag)
    if mmax is not None:
        den -= 10**(-bgr*mmax)
    return numpy.log10(rate / den)


def _weichert_plot(cent_mag, n_obs, binw, t_per, ex_rates_scaled,
                   lcl, ucl, mmax, aval_wei, bval_wei, src_id=None,
                   plt_show=False, ref_mag=None, ref_mag_rate=None,
                   ref_mag_rate_sig=None, bval_sigma=None):

    fig, ax = plt.subplots()

    # Incremental rates of occurrence
    plt.plot(cent_mag, n_obs/t_per, 's', markerfacecolor='none',
             label='Incremental rates')

    alo = [pe.withStroke(linewidth=4, foreground="white")]
    for tm, tn, tp in zip(cent_mag, n_obs, t_per):
        if tn > 0:
            plt.text(tm, tn/tp, f'{tn:.0f}', fontsize=6, path_effects=alo)

    # Rates of exceedance
    plt.plot(cent_mag-binw/2, ex_rates_scaled, 's', markerfacecolor='none',
             color='red', label='Cumulative rates')

    # Rates of exceedance + uncertainty
    fun = _get_gr_double_trunc_exceedance_rates
    if ref_mag is not None:

        # Lower
        alpha = 0.7
        eps_bgr = +1
        eps_rate = -1
        tmp_rate = ref_mag_rate + eps_rate * ref_mag_rate_sig
        tmp_bgr = bval_wei + eps_bgr * bval_sigma
        tmp_agr = _get_agr(tmp_bgr, tmp_rate, ref_mag, mmax=None)
        xmag, exra = fun(tmp_agr, tmp_bgr, cent_mag, binw, mmax)
        lab = f'rate m$_{{{ref_mag:.1f}}}${eps_rate:+.1f}$\sigma$'
        lab += f' bgr{eps_bgr:+.1f}$\sigma$'
        plt.plot(xmag, exra, ls='-.', color='orange', label=lab,
                 alpha=alpha)

        # Upper
        eps_bgr = -1
        eps_rate = +1
        tmp_rate = ref_mag_rate + eps_rate * ref_mag_rate_sig
        tmp_bgr = bval_wei + eps_bgr * bval_sigma
        tmp_agr = _get_agr(tmp_bgr, tmp_rate, ref_mag, mmax=None)
        xmag, exra = fun(tmp_agr, tmp_bgr, cent_mag, binw, mmax)
        lab = f'rate m$_{{{ref_mag:.1f}}}${eps_rate:+.1f}$\sigma$'
        lab += f' bgr{eps_bgr:+.1f}$\sigma$'
        plt.plot(xmag, exra, ls='-.', color='purple', label=lab,
                 alpha=alpha)

        # Lower
        eps_bgr = 2
        eps_rate = -2
        tmp_rate = ref_mag_rate + eps_rate * ref_mag_rate_sig
        tmp_bgr = bval_wei + eps_bgr * bval_sigma
        tmp_agr = _get_agr(tmp_bgr, tmp_rate, ref_mag, mmax=None)
        xmag, exra = fun(tmp_agr, tmp_bgr, cent_mag, binw, mmax)
        lab = f'rate m$_{{{ref_mag:.1f}}}${eps_rate:+.1f}$\sigma$'
        lab += f' bgr{eps_bgr:+.1f}$\sigma$'
        plt.plot(xmag, exra, ls=':', color='orange', label=lab,
                 alpha=alpha)

        # Upper
        eps_bgr = -2
        eps_rate = +2
        tmp_rate = ref_mag_rate + eps_rate * ref_mag_rate_sig
        tmp_bgr = bval_wei + eps_bgr * bval_sigma
        tmp_agr = _get_agr(tmp_bgr, tmp_rate, ref_mag, mmax=None)
        xmag, exra = fun(tmp_agr, tmp_bgr, cent_mag, binw, mmax)
        lab = f'rate m$_{{{ref_mag:.1f}}}${eps_rate:+.1f}$\sigma$'
        lab += f' bgr{eps_bgr:+.1f}$\sigma$'
        plt.plot(xmag, exra, ls=':', color='purple', label=lab,
                 alpha=alpha)

    # Confidence intervals
    plt.plot(cent_mag-binw/2, lcl, '--', color='blue', label='16th C.I.')
    plt.plot(cent_mag-binw/2, ucl, '-.', color='blue', label='84th C.I.')

    xmag, exra = fun(aval_wei, bval_wei, cent_mag, binw, mmax)
    plt.plot(xmag, exra, '--', lw=3, color='green')

    plt.yscale('log')
    plt.xlabel('Magnitude')
    plt.ylabel('Annual rate of exceedance')
    bbox = dict(facecolor='white', alpha=0.8, edgecolor='None')
    plt.text(0.98, 0.95, 'bGR = {:5.3f}'.format(bval_wei),
             transform=ax.transAxes, bbox=bbox, ha='right')
    plt.text(0.98, 0.90, 'aGR = {:5.3f}'.format(aval_wei),
             transform=ax.transAxes, bbox=bbox, ha='right')
    plt.grid(which='major', color='grey')
    plt.grid(which='minor', linestyle='--', color='lightgrey')
    plt.title(src_id)
    plt.legend(fontsize=8, loc=3)

    if plt_show:
        plt.show()

    return fig


def weichert_analysis(fname_input_pattern, fname_config, folder_out=None,
                      folder_out_figs=None, skip=[], binw=0.1,
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
    :param plt_show:
        Boolean. When true show the plots on screen.
    """

    # Create output folders if needed
    if folder_out is not None:
        create_folder(folder_out)
    if folder_out_figs is not None:
        create_folder(folder_out_figs)

    # Parsing config
    if fname_config is not None:
        model = toml.load(fname_config)

    # Set the bin width
    binw = model.get('bin_width', binw)

    # `fname_input_pattern` can be either a list or a pattern (defined by a
    # string)
    if isinstance(fname_input_pattern, str):
        fname_list = list(glob(fname_input_pattern))
    else:
        fname_list = fname_input_pattern

    # Process files with subcatalogues
    for fname in sorted(fname_list):
        print(fname, end='')

        # Get source ID
        src_id = _get_src_id(fname)
        if src_id in skip:
            print("   skipping")
            continue
        else:
            print("")

        # Check if the configuration file there is already information about
        # the current source. Otherwise, use default information to set:
        # - The maximum magnitude (only used while plotting)
        # - The completeness table
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

        # Process catalogue
        tcat = _load_catalogue(fname)
        if tcat is None or len(tcat.data['magnitude']) < 2:
            print('    Source {:s} has less than 2 eqks'.format(src_id))
            continue
        tcat = _add_defaults(tcat)

        # Compute the number of earthquakes per magnitude bin using the
        # completeness table provided
        tcat.data["dtime"] = tcat.get_decimal_time()
        cent_mag, t_per, n_obs = get_completeness_counts(tcat, ctab, binw)

        # When the output folder is defined, save information about eqks count
        if folder_out is not None:
            df = pd.DataFrame()
            df['mag'] = cent_mag
            df['deltaT'] = t_per
            df['nobs'] = n_obs
            fmt = 'occ_count_zone_{:s}'
            fout = os.path.join(folder_out, fmt.format(src_id))
            df.to_csv(fout, index=False)

        # Compute aGR and bGR using Weichert
        out = _weichert_analysis(tcat, ctab, binw, cent_mag, n_obs, t_per)
        aval, bval, lcl, ucl, ex_rat, ex_rts_scl, rmag, rm_rate, rm_sig, sigmab, sigmaa = out

        # Plot
        _weichert_plot(cent_mag, n_obs, binw, t_per, ex_rts_scl,
                       lcl, ucl, mmax, aval, bval, src_id, plt_show)

        # Save results in the configuration file
        if 'sources' not in model:
            model['sources'] = {}
        if src_id not in model['sources']:
            model['sources'][src_id] = {}
        tmp = f"{aval:.5e}"
        model['sources'][src_id]['agr_weichert'] = float(tmp)
        tmp = f"{bval:.5f}"
        model['sources'][src_id]['bgr_weichert'] = float(tmp)
        tmp = f"{rmag:.5e}"
        model['sources'][src_id]['rmag'] = float(tmp)
        tmp = f"{rm_rate:.5e}"
        model['sources'][src_id]['rmag_rate'] = float(tmp)
        tmp = f"{rm_sig:.5e}"
        model['sources'][src_id]['rmag_rate_sig'] = float(tmp)
        tmp = f"{sigmab:.5e}"
        model['sources'][src_id]['bgr_sig'] = float(tmp)
        tmp = f"{sigmaa:.5e}"
        model['sources'][src_id]['agr_sig'] = float(tmp)

        # Save figures
        if folder_out_figs is not None:
            ext = 'png'
            fmt = 'fig_mfd_{:s}.{:s}'
            figure_fname = os.path.join(folder_out_figs,
                                        fmt.format(src_id, ext))
            plt.savefig(figure_fname, format=ext)
            plt.close()

    # Save results the updated config into a file
    if fname_config is not None:
        with open(fname_config, 'w') as f:
            f.write(toml.dumps(model))
