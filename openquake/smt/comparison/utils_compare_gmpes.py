# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.
"""
Module with utility functions for generating trellis plots, response spectra,
hierarchical clustering plots, Sammons maps and Euclidean distance matrix plots
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from scipy import interpolate

from openquake.smt.comparison.sammons import sammon
from openquake.hazardlib.imt import from_string
from openquake.smt.comparison.utils_gmpes import att_curves, _param_gmpes, mgmpe_check


def plot_trellis_util(config, output_directory):
    """
    Generate trellis plots for given run configuration
    """
    # Get mag and dep lists
    mag_list = config.mag_list
    dep_list = config.depth_list
    
    # Median, plus sigma, minus sigma per gmc for up to 4 gmc logic trees
    gmc_p= [[{}, {}, {}], [{}, {}, {}], [{}, {}, {}], [{}, {}, {}]]

    # Get lt weights
    lt_weights = [config.lt_weights_gmc1, config.lt_weights_gmc2,
                  config.lt_weights_gmc3, config.lt_weights_gmc4]
    
    # Get config key
    cfg_key = f'vs30 = {config.vs30} m/s, GMM sigma epsilon = {config.Nstd}'
    
    # Get colours
    colors = get_colors(config.custom_color_flag, config.custom_color_list) 
    
    # Compute attenuation curves
    store_gmm_curves, store_per_imt = {}, {} # For exporting gmm att curves
    store_gmm_curves[cfg_key] = {}
    store_gmm_curves[cfg_key]['gmm att curves per imt-mag'] = {}
    store_gmm_curves[cfg_key]['gmc logic tree curves per imt-mag'] = {}
    fig = pyplot.figure(figsize=(len(mag_list)*5, len(config.imt_list)*4))
    max_pred, min_pred, axs = [], [], []
    for n, i in enumerate(config.imt_list):
        store_per_mag = {}
        for l, m in enumerate(mag_list):
            ax = fig.add_subplot(len(config.imt_list), len(mag_list), l+1+n*len(mag_list))
            axs.append(ax)

            # get ztor
            if config.ztor != -999:
                ztor_m = config.ztor[l]
            else:
                ztor_m = None
            
            # Get gmpe params
            strike_g, dip_g, depth_g, aratio_g = _param_gmpes(config.strike,
                                                              config.dip,
                                                              dep_list[l],
                                                              config.aratio,
                                                              config.rake,
                                                              config.trt) 

            # Per GMPE get attenuation curves
            lt_vals_gmc = [{}, {}, {}, {}]
            store_per_gmpe = {}
            for g, gmpe in enumerate(config.gmpes_list): 
                
                # Sub dicts for median, gmm sigma, median +/- Nstd * gmm sigma
                store_per_gmpe[gmpe] = {}
                col = colors[g]                
                
                # Perform mgmpe check
                gmm = mgmpe_check(gmpe)
                
                # Get attenuation curves
                mean, std, r_vals, tau, phi = att_curves(gmm,
                                                         dep_list[l],
                                                         m,
                                                         aratio_g,
                                                         strike_g,
                                                         dip_g,
                                                         config.rake,
                                                         config.vs30,
                                                         config.z1pt0,
                                                         config.z2pt5,
                                                         config.maxR,
                                                         1, # Step of 1 km for site spacing
                                                         i,
                                                         ztor_m,
                                                         config.dist_type,
                                                         config.trt,
                                                         config.up_or_down_dip,
                                                         config.volc_back_arc,
                                                         config.eshm20_region)

                # Get mean, sigma components, mean plus/minus sigma
                mean = mean[0][0]
                std = std[0][0]
                plus_sigma = np.exp(mean+config.Nstd*std[0])
                minus_sigma = np.exp(mean-config.Nstd*std[0])

                # For managing ylim
                max_pred.append(np.max([np.exp(mean), plus_sigma]))
                min_pred.append(np.min([np.exp(mean), minus_sigma]))

                # Plot predictions and get lt weighted predictions
                lt_vals_gmc = trellis_data(gmpe,
                                           r_vals,
                                           mean,
                                           plus_sigma,
                                           minus_sigma,
                                           col,
                                           config.Nstd,
                                           lt_vals_gmc,
                                           lt_weights)
                
                # Get unit of imt for the store
                unit = get_imtl_unit_for_trellis_store(i)

                # Store per gmpe
                store_per_gmpe[gmpe]['median (%s)' % unit] = np.exp(mean)
                store_per_gmpe[gmpe]['sigma (ln)'] = std
                if config.Nstd != 0:
                    store_per_gmpe[gmpe]['median plus sigma (%s)' % unit] = plus_sigma
                    store_per_gmpe[gmpe]['median minus sigma (%s)' % unit] = minus_sigma
                   
                # Update plots
                update_trellis_plots(m,
                                     i,
                                     n,
                                     l,
                                     config.minR,
                                     config.maxR,
                                     r_vals,
                                     config.imt_list,
                                     config.dist_type)
         
            # Plot logic trees if specified and also store
            for idx_gmc, gmc in enumerate(lt_weights):
                store_gmm_curves = trel_logic_trees(idx_gmc,
                                                    gmc,
                                                    lt_vals_gmc[idx_gmc],
                                                    gmc_p[idx_gmc],
                                                    store_gmm_curves,
                                                    r_vals,
                                                    config.Nstd,
                                                    i,
                                                    m,
                                                    dep_list[l],
                                                    dip_g,
                                                    config.rake,
                                                    cfg_key,
                                                    unit)
                    
            # Store per gmpe
            mag_key = 'Mw = %s, depth = %s km, dip = %s deg, rake = %s deg' % (
                m, dep_list[l], dip_g, config.rake)
            store_per_mag[mag_key] = store_per_gmpe
            pyplot.grid(axis='both', which='both', alpha=0.5)
            
        # Store per imt
        store_per_imt[str(i)] = store_per_mag
    
    # Final store to add vs30 and Nstd into key
    store_gmm_curves[cfg_key][
        'gmm att curves per imt-mag'] = store_per_imt
    store_gmm_curves[cfg_key][
        'gmm att curves per imt-mag']['%s (km)' % config.dist_type] = r_vals
    
    # Finalise plots
    maxy = np.max(max_pred)
    miny = np.min(min_pred)
    for ax in axs: ax.set_ylim(miny, maxy)
    output = os.path.join(output_directory, 'TrellisPlots.png')
    pyplot.legend(loc="center left", bbox_to_anchor=(1.1, 1.05), fontsize='16')
    pyplot.savefig(output, bbox_inches='tight', dpi=200, pad_inches=0.2)
    
    return store_gmm_curves
    

def plot_spectra_util(config, output_directory, obs_spectra_fname):
    """
    Plot response spectra for given run configuration. Can also plot an
    observed spectrum and the corresponding predictions by the specified GMPEs
    """
    # Get mag and depth lists
    mag_list = config.mag_list
    dep_list = config.depth_list

    # If obs spectra csv provided load the data
    if obs_spectra_fname is not None:
        obs_spectra, max_period, eq_id, st_id = load_obs_spectra(obs_spectra_fname)
    else:
        max_period = config.max_period
        obs_spectra, eq_id, st_id = None, None, None
        
    # Get gmc lt weights, imts, periods
    gmc_weights = [config.lt_weights_gmc1, config.lt_weights_gmc2,
                   config.lt_weights_gmc3, config.lt_weights_gmc4]
    imt_list, periods = _get_imts(max_period)
    
    # Get colours and make the figure
    colors = get_colors(config.custom_color_flag, config.custom_color_list)     
    figure = pyplot.figure(figsize=(len(mag_list)*5, len(config.dist_list)*4))
    
    # Set dicts to store values
    dic = {gmm: {} for gmm in config.gmpes_list}  
    lt_vals = [{}, {}, {}, {}]
    lt_vals_plus = [dic, dic, dic, dic]
    lt_vals_minus = [dic, dic, dic, dic]
    store_gmc_lts = []
    
    # Plot the data
    for n, dist in enumerate(config.dist_list):
        for l, m in enumerate(mag_list):
            
            ax1 = figure.add_subplot(
                len(config.dist_list), len(mag_list), l+1+n*len(mag_list))
        
            strike_g, dip_g, depth_g, aratio_g = _param_gmpes(config.strike,
                                                              config.dip,
                                                              dep_list[l],
                                                              config.aratio,
                                                              config.rake,
                                                              config.trt)

            for g, gmpe in enumerate(config.gmpes_list):     
                rs_50p, sig, rs_ps, rs_ms = [], [], [], []
                col = colors[g]
                gmm = mgmpe_check(gmpe)
                
                for k, imt in enumerate(imt_list): 
                    if config.ztor != -999:
                        ztor_m = config.ztor[l]
                    else:
                        ztor_m = None
                        
                    # Get mean and sigma
                    mu, std, r_vals, tau, phi = att_curves(gmm,
                                                           dep_list[l],
                                                           m,
                                                           aratio_g,
                                                           strike_g,
                                                           dip_g,
                                                           config.rake,
                                                           config.vs30,
                                                           config.z1pt0,
                                                           config.z2pt5,
                                                           500, # Assume record dist < 500 km
                                                           1,   # Step of 1 km for site spacing
                                                           imt,
                                                           ztor_m,
                                                           config.dist_type,
                                                           config.trt,
                                                           config.up_or_down_dip,
                                                           config.volc_back_arc,
                                                           config.eshm20_region) 
                    
                    # Interpolate for distances and store
                    mu = mu[0][0]
                    f = interpolate.interp1d(r_vals, mu)
                    try:
                        rs_50p_dist = np.exp(f(dist))
                    except:
                        rtype = config.dist_type
                        assert rtype not in ["repi"] # Should not be interp issues for repi
                        r_min = int(r_vals.min())
                        r_max = int(r_vals.max())
                        raise ValueError(f"Request spectra distance ({rtype} = {dist} km) is "
                                         f"outside of {rtype} value range for this ground-"
                                         f"shaking scenario (min = {r_min} km, max = {r_max} km)")
                    rs_50p.append(rs_50p_dist)
                    
                    f1 = interpolate.interp1d(r_vals, std[0])
                    sigma_dist = f1(dist)
                    sig.append(sigma_dist)
                    
                    if config.Nstd != 0:
                        rs_plus_sigma_dist = np.exp(f(dist)+(config.Nstd*sigma_dist))
                        rs_minus_sigma_dist = np.exp(f(dist)-(config.Nstd*sigma_dist))
                        rs_ps.append(rs_plus_sigma_dist)
                        rs_ms.append(rs_minus_sigma_dist)
                        
                # Plot individual GMPEs
                if 'plot_lt_only' not in str(gmpe):
                    ax1.plot(periods,
                             rs_50p,
                             color=col,
                             linewidth=2,
                             linestyle='-',
                             label=gmpe)
                    if config.Nstd != 0:
                        ax1.plot(periods, rs_ps, color=col, linewidth=0.75, linestyle='-.')
                        ax1.plot(periods, rs_ms, color=col, linewidth=0.75, linestyle='-.')
                
                # Weight the predictions using logic tree weights
                gmc_vals = spectra_data(gmpe,
                                        config.Nstd,
                                        gmc_weights,
                                        rs_50p,
                                        rs_ps,
                                        rs_ms,
                                        lt_vals,
                                        lt_vals_plus,
                                        lt_vals_minus)

                # Plot obs spectra if required
                if obs_spectra is not None:
                    plot_obs_spectra(ax1,
                                     obs_spectra,
                                     g,
                                     config.gmpes_list,
                                     mag_list,
                                     dep_list,
                                     config.dist_list,
                                     eq_id,
                                     st_id)
                
                # Update plots
                update_spec_plots(ax1, m, dist, n, l, config.dist_list, config.dist_type)
            
            # Set axis limits and add grid
            ax1.set_xlim(min(periods), max(periods))
            ax1.grid(True)
            
            # Plot logic trees if required
            for idx_gmc, gmc in enumerate(gmc_weights):
                if gmc_vals[idx_gmc][0] != {}:
                    checks = lt_spectra(ax1,
                                        gmpe,
                                        config.gmpes_list,
                                        config.Nstd,
                                        periods,
                                        idx_gmc,
                                        gmc_vals[idx_gmc][0],
                                        gmc_vals[idx_gmc][1],
                                        gmc_vals[idx_gmc][2])
                    store_gmc_lts.append(checks)    
                
    # Finalise the plots and save fig
    if len(mag_list) * len(config.dist_list) == 1:
        bbox_coo = (1.1, 0.5)
        fs = '10'
    else:
        bbox_coo = (1.1, 1.05)
        fs = '16'
    ax1.legend(loc="center left", bbox_to_anchor=bbox_coo, fontsize=fs)
    save_spectra_plot(figure, obs_spectra, output_directory, eq_id, st_id)

    return store_gmc_lts # Returned for unit tests of gmc lt values


def plot_ratios_util(config, output_directory):
    """
    Generate ratio (GMPE median attenuation/baseline GMPE median attenuation) 
    plots for given run configuration
    """
    # Get mag and dep lists
    mag_list = config.mag_list
    dep_list = config.depth_list

    # Get colours
    colors = get_colors(config.custom_color_flag, config.custom_color_list) 
    
    # Compute ratio curves
    fig = pyplot.figure(figsize=(len(mag_list)*5, len(config.imt_list)*4))
    ratio_store = []
    for n, i in enumerate(config.imt_list):
        for l, m in enumerate(mag_list):
            fig.add_subplot(
                len(config.imt_list), len(mag_list), l+1+n*len(mag_list))
            
            # ztor value
            if config.ztor != -999:
                ztor_m = config.ztor[l]
            else:
                ztor_m = None
            
            # Get gmpe params
            strike_g, dip_g, depth_g, aratio_g = _param_gmpes(config.strike,
                                                              config.dip,
                                                              dep_list[l],
                                                              config.aratio,
                                                              config.rake,
                                                              config.trt) 

            # Load the baseline GMM and compute baseline
            baseline = mgmpe_check(config.baseline_gmm)

            # Get baseline GMM attenuation curves
            results = att_curves(baseline,
                                 dep_list[l],
                                 m,
                                 aratio_g,
                                 strike_g,
                                 dip_g,
                                 config.rake,
                                 config.vs30,
                                 config.z1pt0,
                                 config.z2pt5,
                                 config.maxR,
                                 1, # Step of 1 km for sites
                                 i,
                                 ztor_m,
                                 config.dist_type,
                                 config.trt,
                                 config.up_or_down_dip,
                                 config.volc_back_arc,
                                 config.eshm20_region)
            b_mean = results[0][0][0]

            # Now compute ratios for each GMM
            for g, gmpe in enumerate(config.gmpes_list):        
                
                # Perform mgmpe check
                col = colors[g]         
                gmm = mgmpe_check(gmpe)
                
                # Get attenuation curves for the GMM
                results = att_curves(gmm,
                                     dep_list[l],
                                     m,
                                     aratio_g,
                                     strike_g,
                                     dip_g,
                                     config.rake,
                                     config.vs30,
                                     config.z1pt0,
                                     config.z2pt5,
                                     config.maxR,
                                     1, # Step of 1 km for sites
                                     i,
                                     ztor_m,
                                     config.dist_type,
                                     config.trt,
                                     config.up_or_down_dip,
                                     config.volc_back_arc,
                                     config.eshm20_region)

                # Get mean and r_vals
                mean = results[0][0][0]
                r_vals = results[2]

                # Compute GMM/baseline
                ratio = np.exp(mean)/np.exp(b_mean)
                ratio_store.append(ratio)

                # Plot ratios
                pyplot.semilogy(r_vals,
                                ratio,
                                color=col,
                                linewidth=2, 
                                linestyle='-',
                                label=gmpe)
                
                # Update plots
                update_ratio_plots(config.dist_type,
                                   m,
                                   i,
                                   n,
                                   l,
                                   config.imt_list,
                                   r_vals,
                                   config.minR,
                                   config.maxR)
    
    # Finalise plots
    pyplot.legend(loc="center left", bbox_to_anchor=(1.1, 1.05), fontsize='16')
    out = os.path.join(output_directory, 'RatioPlots.png')
    pyplot.savefig(out, bbox_inches='tight', dpi=200, pad_inches=0.2)


def compute_matrix_gmpes(config, mtxs_type):
    """
    Compute matrix of median ground-motion predictions for each gmpe for the
    given run configuration for use within the Sammon's maps and hierarchical
    clustering dendrograms and Euclidean distance matrix plots
    :param mtxs_type:
        type of predicted ground-motion matrix being computed in
        compute_matrix_gmpes (either median, 84th or 16th percentile)
    """
    # Get mag, imt and depth lists
    mag_list = config.mags_eucl
    dep_list = config.depths_eucl
    imt_list = config.imt_list
    
    mtxs_median = {}
    for n, i in enumerate(imt_list): # Iterate through imt_list
        matrix_medians=np.zeros((len(config.gmpes_list), (len(mag_list)*int((
            config.maxR-config.minR)/1))))

        for g, gmpe in enumerate(config.gmpes_list): 
            medians, sigmas = [], []
            for l, m in enumerate(mag_list): # Iterate though mag_list
            
                gmm = mgmpe_check(gmpe)

                strike_g, dip_g, depth_g, aratio_g = _param_gmpes(config.strike,
                                                                  config.dip,
                                                                  dep_list[l],
                                                                  config.aratio,
                                                                  config.rake,
                                                                  config.trt) 
                
                # ztor              
                if config.ztor != -999:
                    ztor_m = config.ztor[l]
                else:
                    ztor_m = None

                mean, std, r_vals, tau, phi = att_curves(gmm,
                                                         dep_list[l],
                                                         m,
                                                         aratio_g,
                                                         strike_g,
                                                         dip_g,
                                                         config.rake,
                                                         config.vs30,
                                                         config.z1pt0,
                                                         config.z2pt5,
                                                         config.maxR,
                                                         1, # Step of 1 km for site spacing
                                                         i, 
                                                         ztor_m, 
                                                         config.dist_type,
                                                         config.trt,
                                                         config.up_or_down_dip,
                                                         config.volc_back_arc,
                                                         config.eshm20_region) 
                
                # Get means further than minR
                idx = np.argwhere(r_vals>=config.minR).flatten()
                mean = [mean[0][0][idx]]
                std = [std[0][0][idx]]
                tau = [tau[0][0][idx]]
                phi = [phi[0][0][idx]]
                
                if mtxs_type == 'median':
                    medians = np.append(medians, (np.exp(mean)))
                if mtxs_type == '84th_perc':
                    Nstd = 1 # Median + 1std = ~84th percentile
                    medians = np.append(medians, (np.exp(mean+Nstd*std[0])))
                if mtxs_type == '16th_perc':
                    Nstd = 1 # Median - 1std = ~16th percentile
                    medians = np.append(medians, (np.exp(mean-Nstd*std[0])))   
                sigmas = np.append(sigmas, std[0])
                
            matrix_medians[:][g]= medians
        mtxs_median[n] = matrix_medians
        
    return mtxs_median


def plot_euclidean_util(imt_list, gmpe_list, mtxs, namefig, mtxs_type):
    """
    Plot Euclidean distance matrices for given run configuration
    :param imt_list:
        A list e.g. ['PGA', 'SA(0.1)', 'SA(1.0)']
    :param gmpe_list:
        A list e.g. ['BooreEtAl2014', 'CauzziEtAl2014']
    :param mtxs:
        Matrix of predicted ground-motion for each gmpe per imt 
    :param namefig:
        filename for outputted figure 
    :param mtxs_type:
        type of predicted ground-motion matrix being computed in
        compute_matrix_gmpes (either median or 84th or 16th percentile)
    """
    # Euclidean
    matrix_Dist = {}

    # Loop over IMTs
    for n, i in enumerate(imt_list):

        # Get the data matrix
        data = mtxs[n]   
        # Agglomerative clustering
        dist = squareform(pdist(data, 'euclidean'))
        matrix_Dist[n] = dist

    # Create the figure
    ncols = 2
    
    if len(imt_list) < 3:
        nrows = 1
    else:
        nrows = int(np.ceil(len(imt_list) / 2)) 
    
    fig2, axs2 = pyplot.subplots(nrows, ncols)
    fig2.set_size_inches(12, 6*nrows)

    for n, i in enumerate(imt_list):                
        if len(imt_list) < 3:
            ax = axs2[n]
        else:
            ax = axs2[np.unravel_index(n, (nrows, ncols))]           
        ax.imshow(matrix_Dist[n],cmap='gray') 
        
        if mtxs_type == 'median':
            ax.set_title(str(i) + ' (median)', fontsize='14')
        if mtxs_type == '84th_perc':
            ax.set_title(str(i) + ' (84th percentile)', fontsize='14')
        if mtxs_type == '16th_perc':
            ax.set_title(str(i) + ' (16th percentile)', fontsize='14')

        ax.xaxis.set_ticks([n for n in range(len(gmpe_list))])
        ax.xaxis.set_ticklabels(gmpe_list,rotation=40)
        ax.yaxis.set_ticks([n for n in range(len(gmpe_list))])
        ax.yaxis.set_ticklabels(gmpe_list)

    # Remove final plot if not required
    if len(imt_list) >= 3 and len(imt_list)/2 != int(len(imt_list)/2):
        ax = axs2[np.unravel_index(n+1, (nrows, ncols))]
        ax.set_visible(False)

    pyplot.savefig(namefig, bbox_inches='tight', dpi=200, pad_inches=0.2)
    pyplot.tight_layout()        
    
    return matrix_Dist

    
def plot_sammons_util(imt_list,
                      gmpe_list,
                      mtxs,
                      namefig,
                      custom_color_flag,
                      custom_color_list,
                      mtxs_type):
    """
    Plot Sammons maps for given run configuration. The mean of the GMPE
    predictions is also considered.
    :param imt_list:
        A list e.g. ['PGA', 'SA(0.1)', 'SA(1.0)']
    :param gmpe_list:
        A list e.g. ['BooreEtAl2014', 'CauzziEtAl2014']
    :param mtxs:
        Matrix of predicted ground-motion for each gmpe per imt 
    :param namefig:
        filename for outputted figure 
    :param mtxs_type:
        type of predicted ground-motion matrix being computed in
        compute_matrix_gmpes (either median or 84th or 16th percentile)
    """
    # Get mean per imt over the gmpes
    mtxs, gmpe_list = matrix_mean(mtxs, gmpe_list)
    
    # Setup
    colors = get_colors(custom_color_flag, custom_color_list)
    texts = []
    if len(imt_list) < 3:
        nrows = 1
    else:
        nrows = int(np.ceil(len(imt_list)/2)) 
    fig = pyplot.figure()
    fig.set_size_inches(12, 6*nrows)
    
    for n, i in enumerate(imt_list):
        
        data = mtxs[n] # Get the data matrix
        coo, cost = sammon(data, display = 1)
        fig.add_subplot(nrows, 2, n+1)

        for g, gmpe in enumerate(gmpe_list):
            # Get colors and marker
            if g == len(gmpe_list)-1:
                col = 'k'
                marker = 'x'
            else:
                marker = 'o'
                col = colors[g]
            
            # Plot data
            pyplot.plot(coo[g, 0], coo[g, 1], marker, markersize=9, color=col,
                        label=gmpe)
            texts.append(pyplot.text(coo[g, 0]+np.abs(coo[g, 0])*0.02,
                                     coo[g, 1]+np.abs(coo[g, 1])*0.,
                                     gmpe_list[g],
                                     ha='left',
                                     color=col))

        pyplot.title(str(i), fontsize='16')
        if mtxs_type == 'median':
            pyplot.title(str(i) + ' (median)', fontsize='14')
        if mtxs_type == '84th_perc':
            pyplot.title(str(i) + ' (84th percentile)', fontsize='14')
        if mtxs_type == '16th_perc':
            pyplot.title(str(i) + ' (16th percentile)', fontsize='14')
        pyplot.grid(axis='both', which='both', alpha=0.5)

    pyplot.legend(loc="center left", bbox_to_anchor=(1.25, 0.50), fontsize='16')
    pyplot.savefig(namefig, bbox_inches='tight', dpi=200, pad_inches=0.2)
    pyplot.tight_layout()
    
    return coo


def plot_cluster_util(imt_list, gmpe_list, mtxs, namefig, mtxs_type):
    """
    Plot hierarchical clusters for given run configuration. The mean of the
    GMPE predictions is also considered.
    :param imt_list:
        A list e.g. ['PGA', 'SA(0.1)', 'SA(1.0)']
    :param gmpe_list:
        A list e.g. ['BooreEtAl2014', 'CauzziEtAl2014']
    :param mtxs:
        Matrix of predicted ground-motion for each gmpe per imt 
    :param namefig:
        filename for outputted figure 
    :param mtxs_type:
        type of predicted ground-motion matrix being computed in
        compute_matrix_gmpes (either median or 84th or 16th percentile)
    """
    # Get mean per imt over the gmpes
    mtxs, gmpe_list = matrix_mean(mtxs, gmpe_list)
    
    # Setup
    ncols = 2    
    if len(imt_list) < 3:
        nrows = 1
    else:
        nrows = int(np.ceil(len(imt_list) / 2)) 
    matrix_Z = {}
    ymax = [0] * len(imt_list)

    # Loop over IMTs
    for n, i in enumerate(imt_list):

        # Get the data matrix
        data = mtxs[n]

        # Agglomerative clustering
        Z = hierarchy.linkage(
            data, method='ward', metric='euclidean', optimal_ordering=True)
        matrix_Z[n] = Z
        ymax[n] = Z.max(axis=0)[2]

    # Create the figure
    fig, axs = pyplot.subplots(nrows, ncols)
    fig.set_size_inches(12, 6*nrows)

    for n, i in enumerate(imt_list):                
    # Set the axis and title
        if len(imt_list) < 3:
            ax = axs[n]
        else:
            ax = axs[np.unravel_index(n, (nrows, ncols))]       
        
        # Plot dendrogram
        dn1 = hierarchy.dendrogram(
            matrix_Z[n], ax=ax, orientation='right', labels=gmpe_list)
        ax.set_xlabel('Euclidean Distance', fontsize='12')
        if mtxs_type == 'median':
            ax.set_title(str(i) + ' (median)', fontsize='12')
        if mtxs_type == '84th_perc':
            ax.set_title(str(i) + ' (84th percentile)', fontsize='12')
        if mtxs_type == '16th_perc':
            ax.set_title(str(i) + ' (16th percentile)', fontsize='12')
            
    # Remove final plot if not required
    if len(imt_list) >= 3 and len(imt_list)/2 != int(len(imt_list)/2):
        ax = axs[np.unravel_index(n+1, (nrows, ncols))]
        ax.set_visible(False)
    if len(imt_list) == 1:
        axs[1].set_visible(False)
        
    pyplot.savefig(namefig, bbox_inches='tight', dpi=200, pad_inches=0.4)
    pyplot.tight_layout() 
    
    return matrix_Z


### Utils for plots
def get_colors(custom_color_flag, custom_color_list):
    """
    Get list of colors for plots
    """
    colors = [
        'b',     
        'g',
        'r',    
        'c',     
        'm',     
        'y',     
        'k',     
        'm',   
        'gold',  
        'tab:grey', 
        'tab:brown',  
        '#FF5733', 
        '#33FF57', 
        '#FF6347',
        '#800080',
        '#008080',
        '#FFD700',
        '#FF1493',
        '#8A2BE2',
        '#7FFF00',
        '#D2691E',
        '#ADFF2F',
        '#2E8B57',
        '#9932CC',
        '#B22222',
        '#4B0082',
        '#FFFF00',
        '#87CEFA',
        '#00FA9A',
        ]
    
    if custom_color_flag is True:
        return custom_color_list
    else:
        return colors

### Trellis utils
def trellis_data(gmpe,
                 r_vals,
                 mean,
                 plus_sigma,
                 minus_sigma,
                 col,
                 Nstd,
                 lt_vals_gmc,
                 lt_weights):
    """
    Plot predictions of a single GMPE (if required) and compute weighted
    predictions from logic tree(s) (again if required)
    """
    # If plotting not only the logic trees, plot each GMPE
    if 'plot_lt_only' not in str(gmpe): 
        pyplot.plot(r_vals, np.exp(mean), color = col, linewidth=2, linestyle='-', label=gmpe)
        
        # Plot mean with plus/minus sigma too if required
        if Nstd > 0:
            pyplot.plot(r_vals, plus_sigma, linewidth=0.75, color=col, linestyle='-.')
            pyplot.plot(r_vals, minus_sigma, linewidth=0.75, color=col, linestyle='-.')
    
    # Now compute the weighted logic trees
    for idx_gmc, gmc in enumerate(lt_vals_gmc):
        if lt_weights[idx_gmc] is None:
            pass
        elif gmpe in lt_weights[idx_gmc]:
            if lt_weights[idx_gmc][gmpe] is not None:
                if Nstd > 0:
                    lt_vals_gmc[idx_gmc][gmpe] = {
                                'median': np.exp(mean)*lt_weights[idx_gmc][gmpe],
                                'plus_sigma': plus_sigma*lt_weights[idx_gmc][gmpe],
                                'minus_sigma': minus_sigma*lt_weights[idx_gmc][gmpe]}
                else:
                    lt_vals_gmc[idx_gmc][
                        gmpe] = {'median': np.exp(mean)*lt_weights[idx_gmc][gmpe]}
                        
    return lt_vals_gmc


def trel_logic_trees(idx_gmc,
                     gmc,
                     lt_vals_gmc,
                     gmc_p,
                     store_gmm_curves,
                     r_vals,
                     Nstd,
                     i,
                     m,
                     dep,
                     dip,
                     rake,
                     cfg_key,
                     unit):
    """
    Manages plotting of the logic tree attenuation curves and adds them to the
    store of exported attenuation curves 
    """
    # If logic tree provided plot and add to attenuation curve store
    if gmc is not None:
        lt_key = 'gmc logic tree %s' % str(idx_gmc+1)
        
        median, plus_sig, minus_sig = lt_trel(r_vals,
                                              Nstd,
                                              i,
                                              m,
                                              dep,
                                              dip, 
                                              rake,
                                              idx_gmc,
                                              lt_vals_gmc,
                                              gmc_p[0],
                                              gmc_p[1],
                                              gmc_p[2])
        
        store_gmm_curves[cfg_key][
            'gmc logic tree curves per imt-mag'][lt_key] = {}
        store_gmm_curves[cfg_key][
            'gmc logic tree curves per imt-mag'][lt_key]['median (%s)' % unit] = median
        
        if Nstd > 0:
            store_gmm_curves[
                cfg_key]['gmc logic tree curves per imt-mag'][
                    lt_key]['median plus sigma (%s)' % unit] = plus_sig
            store_gmm_curves[
                cfg_key]['gmc logic tree curves per imt-mag'][
                    lt_key]['median minus sigma (%s)' % unit] = minus_sig
    
    return store_gmm_curves


def lt_trel(r_vals,
            Nstd,
            i,
            m,
            dep,
            dip,
            rake,
            idx_gmc,
            lt_vals_gmc,
            median_gmc,
            plus_sig_gmc,
            minus_sig_gmc):
    """
    If required plot spectra from the GMPE logic tree(s)
    """
    # Get colors and strings for checks
    col = ['r', 'b', 'g', 'k'][idx_gmc]
    label = f'Logic Tree {idx_gmc + 1}'

    # Get key describing mag-imt combo and some other event info  
    mk = (f'IMT = {i}, Mw = {m}, depth = {dep} km, '
          f'dip = {dip} deg, rake = {rake} deg')

    # Get logic tree 
    lt_df_gmc = pd.DataFrame(
        lt_vals_gmc, index=[
            'median', 'plus_sigma', 'minus_sigma'])

    lt_median = lt_df_gmc.loc['median'].sum()
    median_gmc[mk] = lt_median

    pyplot.plot(r_vals,
                lt_median,
                linewidth=2,
                color=col,
                linestyle='--',
                label=label,
                zorder=100)

    if Nstd > 0:
        lt_plus = lt_df_gmc.loc['plus_sigma'].sum()
        lt_minus = lt_df_gmc.loc['minus_sigma'].sum()

        plus_sig_gmc[mk] = lt_plus
        minus_sig_gmc[mk] = lt_minus

         # Plot both plus and minus sigma curves
        for sigma_val in [lt_plus, lt_minus]:
            pyplot.plot(r_vals,
                        sigma_val,
                        linewidth=0.75,
                        color=col,
                        linestyle='-.',
                        zorder=100)

    return median_gmc, plus_sig_gmc, minus_sig_gmc


def update_trellis_plots(m, i, n, l, minR, maxR, r_vals, imt_list, dist_type):
    """
    Add titles, axis labels and axis limits to trellis plots
    """
    # Labels
    if dist_type == 'repi':
        label = 'Repi (km)'
    if dist_type == 'rrup':
        label = 'Rrup (km)'
    if dist_type == 'rjb':
        label = 'Rjb (km)'
    if dist_type == 'rhypo':
        label = 'Rhypo (km)'
    if n == 0: # Top row only
        pyplot.title('Mw = ' + str(m), fontsize='16')
    if n == len(imt_list)-1: # Bottom row only
        pyplot.xlabel(label, fontsize='16')
    if l == 0: # Left row only
        if str(i) in ['PGD', 'SDi']:
            pyplot.ylabel(str(i) + ' (cm)', fontsize='16')
        elif str(i) in ['PGV']:
            pyplot.ylabel(str(i) + ' (cm/s)', fontsize='16')
        elif str(i) in ['IA']:
            pyplot.ylabel(str(i) + ' (m/s)', fontsize='16')
        elif str(i) in ['RSD', 'RSD595', 'RSD575', 'RSD2080', 'DRVT']:
            pyplot.ylabel(str(i) + ' (s)', fontsize='16')
        elif str(i) in ['CAV']:
            pyplot.ylabel(str(i) + ' (g-sec)', fontsize='16')
        elif str(i) in ['MMI']:
            pyplot.ylabel(str(i) + ' (MMI)', fontsize='16')
        elif str(i) in ['FAS', 'EAS']:
            pyplot.ylabel(str(i) + ' (Hz)')
        else:
            pyplot.ylabel(str(i) + ' (g)', fontsize='16') # PGA, SA, AvgSA
    
    # xlims (manage this here because if rrup or rjb will be mag dependent)
    min_r_val = min(r_vals[r_vals>=1])
    pyplot.xlim(np.max([min_r_val, minR]), maxR)

    # And make loglog
    pyplot.loglog()
    

def get_imtl_unit_for_trellis_store(i):
    """
    Return a string of the intensity measure type's physical units of
    measurement
    """
    if str(i) in ['PGD', 'SDi']:
        unit = 'cm' # PGD, inelastic spectral displacement
    elif str(i) in ['PGV']:
        unit = 'cm/s' # PGV
    elif str(i) in ['IA']:
        unit = 'm/s' # Arias intensity
    elif str(i) in ['RSD', 'RSD595', 'RSD575', 'RSD2080', 'DRVT']:
        unit = 's' # Relative significant duration, DRVT
    elif str(i) in ['CAV']:
        unit = 'g-sec' # Cumulative absolute velocity
    elif str(i) in ['MMI']:
        unit = 'MMI' # Modified Mercalli Intensity
    elif str(i) in ['FAS', 'EAS']:
        pyplot.ylabel(str(i) + ' (Hz)') # Fourier/Eff. Amp. Spectrum
    else:
        unit = 'g' # PGA, SA, AvgSA

    return unit


### Spectra utils
def _update_period_spacing(period, threshold, spacing, max_period):
    """
    Update period spacing based on maximum period provided.
    """
    period = pd.Series(period)
    if max(period) > threshold:
        for SA in range(0, len(period)):
            if period[SA] > threshold:
                period = period.drop(SA)
        periods_to_re_add = pd.Series(np.arange(1, max_period, spacing))
        period_df = pd.DataFrame({'periods': period,
                                  'periods_to_re_add': periods_to_re_add,
                                  'max_period': max_period})
        return period_df.melt().value.dropna().unique()
    else:
        return period


def _get_period_values_for_spectra_plots(max_period):
    """
    Get list of periods based on maximum period specified in comparison .toml
    :param max_period:
        Maximum period to compute plots for (note an error will be returned if
        this exceeds the maximum spectral period of a GMPE listed in gmpe_list)
    """
    # Set initial periods with constant spacing of 0.1
    period = list(np.round(np.arange(0, max_period, 0.1), 1))
    period.append(max_period)

    # If period extends beyond 1 s reduce interval to 0.2 s
    period = _update_period_spacing(period, 1, 0.2, max_period)
    
    # If period extends beyond 2 s then reduce interval to 0.5 s
    period = _update_period_spacing(period, 2, 0.5, max_period)
    
    # If period extends beyond 5 s then reduce interval to 1 s
    period = _update_period_spacing(period, 5, 1.0, max_period)

    return period


def _get_imts(max_period):
    """
    Convert period floats to imt classes
    """
    # Get periods
    periods = _get_period_values_for_spectra_plots(max_period)
    
    # Convert from float to imt
    period = np.round(periods,1)
    base_SA_string = 'SA(_)'
    imt_list = []
    for imt in range(0,len(period)):
        if imt == 0:
            SA_string = 'PGA'
        else:
            SA_string = base_SA_string.replace('_', str(period[imt]))
        imt_list.append(SA_string)
    for imt in range(0,len(imt_list)):
        imt_list[imt] = from_string(str(imt_list[imt]))
    
    return imt_list, periods


def spectra_data(gmpe,
                 Nstd,
                 gmc_weights,
                 rs_50p,
                 rs_plus_sigma,
                 rs_minus_sigma,
                 lt_vals,
                 lt_vals_plus,
                 lt_vals_minus):
    """
    If required get the logic tree weighted predictions
    """
    for idx_gmc, gmc in enumerate(gmc_weights):
        if gmc_weights[idx_gmc] is None:
            pass
        elif gmpe in gmc_weights[idx_gmc]:
            if gmc_weights[idx_gmc][gmpe] is not None:
                rs_50p_w, rs_plus_sigma_w, rs_minus_sigma_w = {}, {}, {}
                for idx, rs in enumerate(rs_50p):
                    rs_50p_w[idx] = rs_50p[idx]*gmc_weights[idx_gmc][gmpe]
                    if Nstd > 0:
                        rs_plus_sigma_w[idx] = rs_plus_sigma[idx]*gmc_weights[idx_gmc][gmpe]
                        rs_minus_sigma_w[idx] = rs_minus_sigma[idx]*gmc_weights[idx_gmc][gmpe]
    
                # Store the weighted median for the GMPE
                lt_vals[idx_gmc][gmpe] = {'median': rs_50p_w}
                
                # And if Nstd > 0 store these weighted branches too
                if Nstd > 0:
                    lt_vals_plus[idx_gmc][gmpe,'p_sig'] = {'plus_sigma': rs_plus_sigma_w}
                    lt_vals_minus[idx_gmc][gmpe,'m_sig'] = {'minus_sigma': rs_minus_sigma_w}

    gmc1_vals = [lt_vals[0], lt_vals_plus[0], lt_vals_minus[0]]
    gmc2_vals = [lt_vals[1], lt_vals_plus[1], lt_vals_minus[1]]
    gmc3_vals = [lt_vals[2], lt_vals_plus[2], lt_vals_minus[2]]
    gmc4_vals = [lt_vals[3], lt_vals_plus[3], lt_vals_minus[3]]
    
    return gmc1_vals, gmc2_vals, gmc3_vals, gmc4_vals


def lt_spectra(ax1,
               gmpe,
               gmpe_list,
               Nstd,
               period,
               idx_gmc,
               lt_vals_gmc,
               ltv_plus_sig,
               ltv_minus_sig):
    """
    Plot spectra for the GMPE logic tree
    """    
    # Get colors and string for checks
    colours = ['r', 'b', 'g', 'k']
    col = colours[idx_gmc]
    check = 'lt_weight_gmc' + str(idx_gmc+1)
    label = 'Logic Tree ' + str(idx_gmc+1)
    
    # Plot   
    lt_per_imt_gmc, lt_plus_sig_per_imt, lt_minus_sig_per_imt = {}, {}, {}
    lt_df_gmc = pd.DataFrame(lt_vals_gmc, index=['median'])
    if Nstd > 0:
        lt_plus_sig = pd.DataFrame(ltv_plus_sig, index=['plus_sigma'])
        lt_minus_sig = pd.DataFrame(ltv_minus_sig, index=['minus_sigma'])    
    wt_per_gmpe_gmc, wt_plus_sig, wt_minus_sig = {}, {}, {}
    for gmpe in gmpe_list:
        if check in str(gmpe):
            wt_per_gmpe_gmc[gmpe] = np.array(pd.Series(
                lt_df_gmc[gmpe].loc['median']))
            if Nstd > 0:
                wt_plus_sig[gmpe] = np.array(
                    pd.Series(lt_plus_sig[gmpe,'p_sig'].loc['plus_sigma']))
                wt_minus_sig[gmpe] = np.array(
                    pd.Series(lt_minus_sig[gmpe,'m_sig'].loc['minus_sigma']))
        
    lt_df_gmc = pd.DataFrame(wt_per_gmpe_gmc, index=period)
    lt_plus_sig = pd.DataFrame(wt_plus_sig, index=period)
    lt_minus_sig = pd.DataFrame(wt_minus_sig, index=period)
    for idx, imt in enumerate(period):
        lt_per_imt_gmc[imt] = np.sum(lt_df_gmc.loc[imt])
        if Nstd > 0:
            lt_plus_sig_per_imt[imt] = np.sum(lt_plus_sig.loc[imt])
            lt_minus_sig_per_imt[imt] = np.sum(lt_minus_sig.loc[imt])
    
    # Plot logic tree
    ax1.plot(period,
             pd.Series(lt_per_imt_gmc).values,
             linewidth=2,
             color=col,
             linestyle='--',
             label=label,
             zorder=100)
    
    # Plot mean plus sigma and mean minus sigma if required
    if Nstd > 0:
        
        ax1.plot(period,
                 pd.Series(lt_plus_sig_per_imt).values,
                 linewidth=0.75,
                 color=col,
                 linestyle='-.',
                 zorder=100)   
          
        ax1.plot(period,
                 pd.Series(lt_minus_sig_per_imt).values,
                 linewidth=0.75,
                 color=col,
                 linestyle='-.',
                 zorder=100)
        
    return [lt_per_imt_gmc, lt_plus_sig_per_imt, lt_minus_sig_per_imt]
        
   
def load_obs_spectra(obs_spectra_fname):
    """
    If an obs spectra file has been specified get values from the csv
    for comparison of observed spectra and spectra computed using GMPE
    predictions.
    
    Returns the spectra as a dataframe, the max period of the spectra,
    the earthquake ID and the station ID.
    """
    # Load the obs spectra
    obs_spectra = pd.read_csv(obs_spectra_fname)

    # Get values from obs_spectra dataframe...
    eq_id = str(obs_spectra['EQ ID'].iloc[0])
    st_id = str(obs_spectra['Station Code'].iloc[0])
    
    max_period = obs_spectra['Period (s)'].max()
    
    return obs_spectra, max_period, eq_id, st_id


def plot_obs_spectra(ax1,
                     obs_spectra,
                     g,
                     gmpe_list,
                     mag_list,
                     dep_list,
                     dist_list,
                     eq_id,
                     st_id):
    """
    Check if an observed spectra must be plotted, and if so plot
    """
    # Plot an observed spectra if inputted...
    if obs_spectra is not None and g == len(gmpe_list)-1:
        
        # Get rup params
        mw = np.asarray(mag_list, float)
        rrup = np.asarray(dist_list, float)
        depth = np.asarray(dep_list, float)
        
        # Get label for spectra plot
        obs_string = (eq_id + '\nrecorded at ' + st_id + ' (Rrup = '
                      + str(rrup) + ' km, ' + '\nMw = ' + str(mw) +
                      ', depth = ' + str(depth) + ' km)')
                      
        # Plot the observed spectra
        ax1.plot(obs_spectra['Period (s)'],
                 obs_spectra['SA (g)'],
                 color='r',
                 linewidth=3,
                 linestyle='-',
                 label=obs_string)    
        
        
def update_spec_plots(ax1, m, i, n, l, dist_list, dist_type):
    """
    Add titles and axis labels to spectra plots
    """
    ax1.set_title(f'Mw = {m}, {dist_type} = {i} km', fontsize=16, y=1.0, pad=-16)
    if n == len(dist_list)-1: # Bottom row only
        ax1.set_xlabel('Period (s)', fontsize=16)
    if l == 0: # Left column only
        ax1.set_ylabel('SA (g)', fontsize=16) 


def save_spectra_plot(f1, obs_spectra, output_dir, eq_id, st_id):
    """
    Save the plotted response spectra
    """
    if obs_spectra is None:
        out = os.path.join(output_dir, 'ResponseSpectra.png')
        f1.savefig(out, bbox_inches='tight', dpi=200, pad_inches=0.2)
    else:
        rec_str = str(eq_id) + '_recorded_at_' + str(st_id)
        rec_str = rec_str.replace(' ', '_').replace('-', '_').replace(':', '_')
        out = os.path.join(output_dir, 'ResponseSpectra_' + rec_str + '.png')
        f1.savefig(out, bbox_inches='tight', dpi=200, pad_inches=0.2)


### Utils for other plots
def update_ratio_plots(dist_type, m, i, n, l, imt_list, r_vals, minR, maxR):
    """
    Add titles and axis labels to ratio plots
    """
    if dist_type == 'repi':
        label = 'Repi (km)'
    if dist_type == 'rrup':
        label = 'Rrup (km)'
    if dist_type == 'rjb':
        label = 'Rjb (km)'
    if dist_type == 'rhypo':
        label = 'Rhypo (km)'
    if n == 0: # Top row only
        pyplot.title('Mw = ' + str(m), fontsize='16')
    if n == len(imt_list)-1: # Bottom row only
        pyplot.xlabel(label, fontsize='12')
    if l == 0: # Left row only
        pyplot.ylabel('GMM/baseline for %s' %str(i), fontsize='14')
    min_r_val = min(r_vals[r_vals>=1])
    pyplot.xlim(np.max([min_r_val, minR]), maxR)


def matrix_mean(mtxs, gmpe_list):
    """
    For a matrix of predicted ground-motions compute the arithmetic mean
    per prediction per IMT w.r.t. the number of GMPEs within the gmpe_list
    """
    for _, imt in enumerate(mtxs):
        store_vals = []
        for idx_gmpe, _ in enumerate(mtxs[imt]):
            store_vals.append(
                pd.Series(mtxs[imt][idx_gmpe])) # store per gmpe per imt 
        
        # Into df and get mean val per column (one column per prediction)
        val_df = pd.DataFrame(store_vals)
        mean = (val_df.mean(axis=0)) 
        mtxs[imt] = np.concatenate((mtxs[imt], [mean]))
    
    if not any(x == 'mean' for x in gmpe_list):
        gmpe_list.append('mean')

    return mtxs, gmpe_list