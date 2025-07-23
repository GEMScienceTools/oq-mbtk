import toml
from glob import glob
import numpy as np
import pandas as pd
import os
from scipy.stats import poisson
import matplotlib.pyplot as plt
from openquake.mbt.tools.model_building.dclustering import _add_defaults
from openquake.hmtk.seismicity.occurrence.utils import get_completeness_counts
from openquake.wkf.utils import _get_src_id, create_folder, get_list

# get_mag_year_from_comp_table and trim_eq_catalog_with_completeness_table come from hamlet,
# with some minor modifications to work more directly with mbtk output

def get_mag_year_from_comp_table(comp_table, mag):
    
    yrs = np.array([c[0] for c in comp_table])
    mags = np.array([c[1] for c in comp_table])
    
    next_smaller_mag_idx = np.where(mags <= mag)[0][-1]
    mag = mags[next_smaller_mag_idx]
    comp_year = yrs[next_smaller_mag_idx]

    return mag, comp_year

def trim_eq_catalog_with_completeness_table(
    eq_gdf, comp_table, stop_date, trim_to_completeness=True
):
    
    out_gdf = eq_gdf.loc[eq_gdf.year <= stop_date]
    drop_idxs = []
    mags = np.array([c[1] for c in comp_table])
    
    for i, eq in out_gdf.iterrows():
        try:
            _, comp_year = get_mag_year_from_comp_table(
                comp_table, eq.magnitude
            )
            if eq.year < comp_year:
                drop_idxs.append(i)
        except:
            if trim_to_completeness:
                drop_idxs.append(i)
            else:
                pass

    out_gdf = out_gdf.drop(drop_idxs)
    return out_gdf

def plot_GR_inc_fixedparams_completeness_imp(cat,mbin, a, b, comptab, plt_show = True, plt_title = ''):
    '''
    Given an earthquake catalogue, estimates of the a and b-value and a completeness table, 
    plot the observed and expected number of events in each bin. Expected number includes Poisson
    lower (5%, orange) and upper (95%, blue) bounds

    :param cat:
        catalogue (geo)dataframe with hmtk column names
    :param mbin:
        binwidth for plots
    :param a:
        Gutenberg-Richter a-value estimated for this catalogue (given the completeness)
    :param b:
        Gutenberg-Richter b-value estimated for this catalogue (given the completeness)
    :param comptab:
        numpy array describing completeness upon which GR estimates are based
        e.g. comptab = [[1975,   5.5], [1960, 5.0], [1900, 7.0]]
    :param plt_show:
        boolean specifying if plot should be displayed. Defaults to True if unspecified
    :param plt_title:,
        title for plot

    '''
    
    # set mmin to lowest in completeness windows
    mmin = min(np.array([c[1] for c in comptab]))
    
    # Filter observed catalogue for completeness
    maxyear = max(cat.year)
    comp_cat = trim_eq_catalog_with_completeness_table(cat, comptab, maxyear )
    mags = comp_cat.magnitude[comp_cat.magnitude > mmin-(mbin/2)]
     
    m_bins = np.arange(mmin, max(mags) +0.5, mbin) 
    nbins = len(m_bins)
    comp_years_m = np.zeros(nbins)    
    inc_obs = np.zeros(nbins)
    cum_obs = np.zeros(nbins)
    inc_fit = np.zeros(nbins)
    cum_fit = np.zeros(nbins)
    
    # For each bin, count the number of observed events, 
    # calculate the (cumulative) number of expected events given a, b
    # determine how long this bin has been complete for

    for i in range(nbins):
        cum_obs[i] = len(mags[mags > m_bins[i]-mbin/2])
        cum_fit[i] = (10**(a - b*(m_bins[i])))
        mag, comp_year = get_mag_year_from_comp_table(comptab, m_bins[i],)
        comp_years_m[i] = maxyear - comp_year
    
    # get incremental counts from cumulative    
    inc_obs = np.absolute(np.diff(np.concatenate((cum_obs, [0]),axis=0)))
    inc_fit = np.absolute(np.diff(np.concatenate((cum_fit, [0]),axis=0)))
    
    # Scale incremental expected counts by the number of years of completeness
    inc_fit = inc_fit*comp_years_m
    
    # Make the plot
    fig, ax = plt.subplots()
    
    # plot observed numbers
    ax.scatter(m_bins, inc_obs,  c='black', label = "observed events >= Mc")    
    
    # plot expected (remove last bin, which will be problematic due to 
    # calculation from cumulative)
    plt.plot(m_bins[:-1], inc_fit[:-1], '--',  label = "expected count | completeness", c = "gray")
    
    # calculate and plot poisson count errors
    nhi = poisson.ppf(0.975, inc_fit)
    nlo = poisson.ppf(0.025, inc_fit)
    line2, = ax.plot(m_bins[:-1], nhi[:-1], dashes=[6, 2], c = "blue", label = '95% confidence interval')
    line3, = ax.plot(m_bins[:-1], nlo[:-1], dashes=[6, 2], c = "blue")
    
    plt.xlabel("Magnitude")
    plt.ylabel("Count")
    plt.grid(which='major', color='grey')
    plt.grid(which='minor', linestyle='--', color='lightgrey')
    plt.title(plt_title)
    plt.yscale('log')   
    plt.ylim(bottom = 0.9)
    plt.legend()

    if plt_show:
        plt.show()

    return fig   

def plot_incremental_mfds(fname_input_pattern, fname_config, 
                      folder_out_figs=None, skip=[], binw=0.1,
                      plt_show=False):
    """
    Given a catalogue and a config, plots the incremental number of observed earthquakes within completeness
    windows and the expected counts determined from completeness and fmd parametrs.

    :param fname_input_pattern:
        It can be either a string (definining a pattern) or a list of
        .csv files. The file names must have the source ID at the end. The
        delimiter of the source ID on the left is `_`
    :param fname_config:
        The name of the .toml configuration file
    :param folder_out_figs:
        The folder where to store the figures
    :param skip:
        A list with the IDs of the sources to skip
    :param plt_show:
        Boolean. When true show the plots on screen.
    """

    # Create output folders if needed
    if folder_out_figs is not None:
        create_folder(folder_out_figs)

    # Parsing config
    if fname_config is not None:
        model = toml.load(fname_config)

    # Set the bin width
    #binw = model.get('bin_width', binw)
    binw = float(binw)

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

        # Get completeness, agr and bgr values from config
        if 'sources' in model:
            ctab = np.array(model['sources'][src_id]['completeness_table'])
            aval = model['sources'][src_id]['agr']
            bval = model['sources'][src_id]['bgr']
            #mmax = model['sources'][src_id]['mmax']
        
        # Process catalogue
        tcat = pd.read_csv(fname)
        if tcat is None or len(tcat['magnitude']) < 2:
            print('    Source {:s} has less than 2 eqks'.format(src_id))
            continue
        

        # Plot
        plot_GR_inc_fixedparams_completeness_imp(tcat, binw, aval, bval, ctab, plt_show, src_id)

        # Save figures
        if folder_out_figs is not None:
            ext = 'png'
            fmt = 'fig_inc_comp_{:s}.{:s}'
            figure_fname = os.path.join(folder_out_figs,
                                        fmt.format(src_id, ext))
            plt.savefig(figure_fname, format=ext)
            plt.close()
