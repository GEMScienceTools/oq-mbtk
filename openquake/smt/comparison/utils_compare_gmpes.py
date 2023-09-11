# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2023 GEM Foundation
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
Module with utility functions for generating trellis plots, hierarchical
clustering plots, Sammons maps and Euclidean distance matrix plots
"""
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from scipy import interpolate
from IPython.display import display
from collections import OrderedDict

from openquake.smt.comparison.sammons import sammon
from openquake.hazardlib import valid
from openquake.hazardlib.imt import from_string
from openquake.smt.comparison.utils_gmpes import att_curves, _get_z1,\
    _get_z25, _param_gmpes, mgmpe_check


def plot_trellis_util(
        trt, ztor, rake, strike, dip, depth, Z1, Z25, Vs30, region, imt_list,
        mag_list, maxR, gmpe_list, aratio, Nstd, output_directory,
        custom_color_flag, custom_color_list, eshm20_region, dist_type,
        lt_weights_gmc1 = None, lt_weights_gmc2 = None, up_or_down_dip = None):
    """
    Generate trellis plots for given run configuration
    """
    # Setup
    fig = pyplot.figure(figsize=(len(mag_list)*5, len(imt_list)*4))
    mean_gmc1, plus_sig_gmc1, minus_sig_gmc1 = {}, {}, {}
    mean_gmc2, plus_sig_gmc2, minus_sig_gmc2 = {}, {}, {}
    
    Z1, Z25 = get_z1_z25(Z1, Z25, Vs30, region)
    colors = get_cols(custom_color_flag, custom_color_list)     
    step = 1
    for n, i in enumerate(imt_list):
        for l, m in enumerate(mag_list):
            fig.add_subplot(len(imt_list), len(mag_list), l+1+n*len(mag_list))
            lt_vals_gmc1, lt_vals_gmc2 = {}, {}
            for g, gmpe in enumerate(gmpe_list): 
                
                # Perform mgmpe check
                col = colors[g]
                gsim = valid.gsim(gmpe)
                gmm_orig = gsim
                gmm = mgmpe_check(gsim)
                
                # ZTOR value
                if ztor is not None:
                    ztor_m = ztor[l]
                else:
                    ztor_m = None
                
                # Get gmpe params
                strike_g, dip_g, depth_g, aratio_g = _param_gmpes(
                    strike, dip, depth[l], aratio, rake, trt) 
                
                # Get attenuation curves
                mean, std, r_vals = att_curves(gmm, gmm_orig, depth[l],m, 
                                               aratio_g, strike_g, dip_g,
                                               rake,Vs30, Z1, Z25, maxR, 
                                               step, i, ztor_m, eshm20_region,
                                               dist_type, trt, up_or_down_dip)
                
                # Get mean, sigma, mean plus sigma and mean minus sigma
                mean = mean[0][0]
                std = std[0][0]
                plus_sigma = np.exp(mean+Nstd*std[0])
                minus_sigma = np.exp(mean-Nstd*std[0])
                if 'plot_lt_only' not in str(gmpe): # If not plotting lt only
                    pyplot.plot(r_vals, np.exp(mean), color = col, linewidth=2,
                                linestyle='-', label=gmpe)
                
                # Plot and get lt weighted predictions
                lt_vals_gmc1, lt_vals_gmc2 = trellis_data(
                    Nstd, gmpe, r_vals, mean, plus_sigma, minus_sigma, col, i, m,
                    lt_weights_gmc1, lt_vals_gmc1, lt_weights_gmc2, lt_vals_gmc2)
                
                # update plots
                update_trellis_plots(m, i, n, l, r_vals, imt_list, dist_type)

            pyplot.grid(axis='both', which='both', alpha=0.5)
        
            ### Plot logic trees if specified
            lt_trel(r_vals, Nstd, i, m,
                    lt_vals_gmc1, mean_gmc1, plus_sig_gmc1, minus_sig_gmc1,
                    lt_vals_gmc2, mean_gmc2, plus_sig_gmc2, minus_sig_gmc2)
            
    # Finalise plots
    pyplot.legend(loc = "center left", bbox_to_anchor = (1.1, 1.05),
                  fontsize = '16')
    pyplot.savefig(os.path.join(output_directory, 'TrellisPlots.png'),
                   bbox_inches='tight', dpi=200, pad_inches=0.2)
    

def plot_spectra_util(trt, ztor, rake, strike, dip, depth, Z1, Z25, Vs30,
                      region, max_period, mag_list, dist_list, gmpe_list,
                      aratio, Nstd, output_directory, custom_color_flag,
                      custom_color_list, eshm20_region, dist_type,
                      lt_weights_gmc1 = None, lt_weights_gmc2 = None,
                      obs_spectra = None, up_or_down_dip = None):
    """
    Plot response spectra and sigma w.r.t. spectral period for given run
    configuration. Can also plot an observed spectrum and the corresponding
    predictions by the specified GMPEs
    """
    # If obs_spectra get info from csv
    if obs_spectra is not None:
        eq_id, mw, dep, rrup, st_id, mag_list, dist_list, depth, strike, dip,\
        rake, vs30, ztor, trt, up_or_down_dip = load_obs_spectra(obs_spectra)
    else:
        mw, rrup, dep, eq_id, st_id = None, None, None, None, None
        
    # Get the periods to plot
    period = _get_period_values_for_spectra_plots(max_period)
    
    # Convert periods from floats to imts
    imt_list = _get_imts(period)
    
    # Setup    
    Z1, Z25 = get_z1_z25(Z1, Z25, Vs30, region)
    colors = get_cols(custom_color_flag, custom_color_list)     
    fig1 = pyplot.figure(figsize = (len(mag_list)*5, len(dist_list)*4))
    fig2 = pyplot.figure(figsize = (len(mag_list)*5, len(dist_list)*4))
    
    ### Set dicts to store values
    dic = OrderedDict([(gmm, {}) for gmm in gmpe_list])  
    lt_vals_gmc1, lt_vals_gmc2 = {}, {}
    lt_vals_plus_sig_gmc1, lt_vals_minus_sig_gmc1 = dic, dic
    lt_vals_plus_sig_gmc2, lt_vals_minus_sig_gmc2 = dic, dic

    ### Plot the data
    for n, i in enumerate(dist_list):
        for l, m in enumerate(mag_list):
            
            ax1 = fig1.add_subplot(len(dist_list), len(mag_list), l+1+n*len(
                mag_list))
            ax2 = fig2.add_subplot(len(dist_list), len(mag_list), l+1+n*len(
                mag_list))
            
            for g, gmpe in enumerate(gmpe_list):     
                col = colors[g]
                gsim = valid.gsim(gmpe)
                gmm_orig = gsim
                gmm = mgmpe_check(gsim)
                strike_g, dip_g, depth_g, aratio_g = _param_gmpes(strike, dip,
                                                                  depth[l],
                                                                  aratio, rake,
                                                                  trt)
                
                rs_50p, rs_plus_sigma, rs_minus_sigma, sigma = [], [], [], []
                
                for k, imt in enumerate(imt_list): 
                    if obs_spectra is not None:
                        dist = 1000 # Set to 1000 km
                        Vs30 = vs30 # Set to vs30 in obs_spectra
                        if i > 1000:
                            raise ValueError('Rrup provided for the observed\
                                             spectra is greater than 1000 km')
                    else:
                        dist = i
                    
                    # ZTOR
                    if ztor is not None:
                        ztor_m = ztor[l]
                    else:
                        ztor_m = None
                    
                    # Get mean and sigma
                    mu, std, r_vals = att_curves(gmm, gmm_orig, depth[l], m,
                                                 aratio_g, strike_g, dip_g, 
                                                 rake, Vs30, Z1, Z25, dist,
                                                 0.1, imt, ztor_m, eshm20_region,
                                                 dist_type, trt, up_or_down_dip) 
                    mu = mu[0][0]
                    f = interpolate.interp1d(r_vals, mu)
                    rs_50p_dist = np.exp(f(i))
                    rs_50p.append(rs_50p_dist)
                    f1 = interpolate.interp1d(r_vals, std[0])
                    sigma_dist = f1(i)
                    sigma.append(sigma_dist)
                    
                    if Nstd != 0:
                            rs_plus_sigma_dist = np.exp(f(i)+(Nstd*sigma_dist))
                            rs_minus_sigma_dist = np.exp(f(i)-(Nstd*sigma_dist))
                   
                    if Nstd != 0:
                        rs_plus_sigma.append(rs_plus_sigma_dist)
                        rs_minus_sigma.append(rs_minus_sigma_dist)
                    
                # Plot individual GMPEs
                if 'plot_lt_only' not in str(gmpe):
                    ax1.plot(period, rs_50p, color = col, linewidth=2,
                             linestyle='-', label=gmpe)
                    if Nstd != 0:
                        ax1.plot(period, rs_plus_sigma, color=col,
                                 linewidth=0.75, linestyle='-.')
                        ax1.plot(period, rs_minus_sigma, color=col,
                                 linewidth=0.75, linestyle='-.')
                
                # Plot sigma vs period
                ax2.plot(period, sigma, color=col, linewidth=2, linestyle='-',
                         label=gmpe)
                    
                # Weight the predictions using logic tree weights
                lt_vals_gmc1, lt_vals_plus_sig_gmc1, lt_vals_minus_sig_gmc1, \
                lt_vals_gmc2, lt_vals_plus_sig_gmc2, lt_vals_minus_sig_gmc2 =\
                    spectra_data(gmpe, Nstd,
                                 rs_50p, rs_plus_sigma, rs_minus_sigma,
                                 lt_vals_gmc1, lt_vals_gmc2,
                                 lt_weights_gmc1, lt_weights_gmc2,
                                 lt_vals_plus_sig_gmc1, lt_vals_minus_sig_gmc1,
                                 lt_vals_plus_sig_gmc2, lt_vals_minus_sig_gmc2)

                # Plot obs spectra if required
                plot_obs_spectra(ax1, obs_spectra, g, gmpe_list, mw, dep, rrup)
                
                # Update plots
                update_spec_plots(ax1, ax2, m, i, n, l, dist_list)
            
            # Set axis limits and add grid
            ax1.set_xlim(min(period), max(period))
            ax2.set_ylim(0.3, 1)
            ax1.grid(True)
            ax2.grid(True)
            
            # Plot logic trees if required
            lt_spectra(ax1, gmpe, gmpe_list, Nstd, period, 'gmc1',
                lt_vals_gmc1, lt_vals_plus_sig_gmc1, lt_vals_minus_sig_gmc1)
            
            lt_spectra(ax1, gmpe, gmpe_list, Nstd, period, 'gmc2',
                lt_vals_gmc2, lt_vals_plus_sig_gmc2, lt_vals_minus_sig_gmc2)
                
    # Finalise the plots and save fig
    if len(mag_list) * len(dist_list) == 1:
        bbox_coo = (1.1, 0.5)
        fs = '10'
    else:
        bbox_coo = (1.1, 1.05)
        fs = '16'
    ax1.legend(loc="center left", bbox_to_anchor=bbox_coo, fontsize=fs)
    ax2.legend(loc="center left", bbox_to_anchor=bbox_coo, fontsize=fs)
    save_spectra_plot(fig1, fig2, obs_spectra, output_directory, eq_id, st_id)


def compute_matrix_gmpes(trt, ztor, imt_list, mag_list, gmpe_list, rake, strike,
                         dip, depth, Z1, Z25, Vs30, region, maxR, aratio,
                         eshm20_region, dist_type, mtxs_type, up_or_down_dip):
    """
    Compute matrix of median ground-motion predictions for each gmpe for the
    given run configuration for use within Euclidean distance matrix plots,
    Sammons Mapping and hierarchical clustering plots
    :param mtxs_type:
        type of predicted ground-motion matrix being computed in
        compute_matrix_gmpes (either median, 84th or 16th percentile)
    """
    # Setup
    mtxs_median = {}
    step = 1
    Z1, Z25 = get_z1_z25(Z1, Z25, Vs30, region)
    for n, i in enumerate(imt_list): #iterate though imt_list
        matrix_medians=np.zeros((len(gmpe_list), (len(mag_list)*int((maxR/step)))))

        for g, gmpe in enumerate(gmpe_list): 
            medians, sigmas = [], []
            for l, m in enumerate(mag_list): #iterate though mag_list
                
                gsim = valid.gsim(gmpe)
                gmm_orig = gsim
                gmm = mgmpe_check(gsim)

                strike_g, dip_g, depth_g, aratio_g = _param_gmpes(
                    strike, dip, depth[l], aratio, rake, trt) 
                
                # ZTOR                
                if ztor is not None:
                    ztor_m = ztor[l]
                else:
                    ztor_m = None

                mean, std, r_vals = att_curves(gmm, gmm_orig, depth[l], m, 
                                                  aratio_g, strike_g, dip_g, 
                                                  rake, Vs30, Z1, Z25, maxR, 
                                                  step, i, ztor_m, eshm20_region,
                                                  dist_type, trt, up_or_down_dip) 
                
                if mtxs_type == 'median':
                    medians = np.append(medians, (np.exp(mean)))
                if mtxs_type == '84th_perc':
                    Nstd = 1 # median + 1std = ~84th percentile
                    medians = np.append(medians, (np.exp(mean+Nstd*std[0])))
                if mtxs_type == '16th_perc':
                    Nstd = 1 # median - 1std = ~16th percentile
                    medians = np.append(medians, (np.exp(mean-Nstd*std[0])))   
                sigmas = np.append(sigmas,std[0])

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
            ax.set_title(str(i) + ' (median)', fontsize = '14')
        if mtxs_type == '84th_perc':
            ax.set_title(str(i) + ' (84th percentile)', fontsize = '14')
        if mtxs_type == '16th_perc':
            ax.set_title(str(i) + ' (16th percentile)', fontsize = '14')

        ax.xaxis.set_ticks([n for n in range(len(gmpe_list))])
        ax.xaxis.set_ticklabels(gmpe_list,rotation=40)
        ax.yaxis.set_ticks([n for n in range(len(gmpe_list))])
        ax.yaxis.set_ticklabels(gmpe_list)

    # Remove final plot if not required
    if len(imt_list) > 3 and len(imt_list)/2 != int(len(imt_list)/2):
        ax = axs2[np.unravel_index(n+1, (nrows, ncols))]
        ax.set_visible(False)

    pyplot.savefig(namefig, bbox_inches = 'tight', dpi = 200, pad_inches = 0.2)
    pyplot.tight_layout()        
    
    return matrix_Dist

    
def plot_sammons_util(imt_list, gmpe_list, mtxs, namefig, custom_color_flag,
                      custom_color_list, mtxs_type):
    """
    Plot Sammons maps for given run configuration
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
    # Setup
    colors = get_cols(custom_color_flag, custom_color_list)
    texts = []
    
    ncols = 2    
    if len(imt_list) < 3:
        nrows = 1
    else:
        nrows = int(np.ceil(len(imt_list) / 2)) 
    
    fig = pyplot.figure()
    fig.set_size_inches(12, 6*nrows)
    
    for n, i in enumerate(imt_list): #iterate though imt_list

        # Get the data matrix
        data = mtxs[n]
        coo, cost = sammon(data, display = 1)
        
        fig.add_subplot(nrows, ncols, n+1) #(#vert, #hor, #subplot)

        for g, gmpe in enumerate(gmpe_list): 
            col=colors[g]
            pyplot.plot(coo[g, 0], coo[g, 1], 'o', markersize=9, color=colors[
                g], label=gmpe)
            texts.append(pyplot.text(coo[g, 0]+np.abs(coo[g, 0])*0.02,
                                     coo[g, 1]+np.abs(coo[g, 1])*0.,
                                     gmpe_list[g], ha = 'left', color = col))

        pyplot.title(str(i), fontsize='16')
        if mtxs_type == 'median':
            pyplot.title(str(i) + ' (median)', fontsize = '14')
        if mtxs_type == '84th_perc':
            pyplot.title(str(i) + ' (84th percentile)', fontsize = '14')
        if mtxs_type == '16th_perc':
            pyplot.title(str(i) + ' (16th percentile)', fontsize = '14')
        pyplot.grid(axis='both', which='both', alpha=0.5)

    pyplot.legend(loc = "center left", bbox_to_anchor = (1.25, 0.50),
                  fontsize = '16')
    pyplot.savefig(namefig, bbox_inches = 'tight', dpi = 200, pad_inches = 0.2)
    pyplot.tight_layout()
    
    return coo


def plot_cluster_util(imt_list, gmpe_list, mtxs, namefig, mtxs_type):
    """
    Plot hierarchical clusters for given run configuration
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
    # Setup
    ncols = 2    
    if len(imt_list) < 3:
        nrows = 1
    else:
        nrows = int(np.ceil(len(imt_list) / 2)) 
    matrix_Z = {}
    ymax =  [0] * len(imt_list)

    # Loop over IMTs
    for n, i in enumerate(imt_list):

        # Get the data matrix
        data = mtxs[n]

        # Agglomerative clustering
        Z = hierarchy.linkage(data, method = 'ward', metric = 'euclidean',
                              optimal_ordering = True)
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
        dn1 = hierarchy.dendrogram(matrix_Z[n], ax=ax, orientation='right',
                                   labels=gmpe_list)
        ax.set_xlabel('Euclidean Distance', fontsize = '12')
        if mtxs_type == 'median':
            ax.set_title(str(i) + ' (median)', fontsize = '12')
        if mtxs_type == '84th_perc':
            ax.set_title(str(i) + ' (84th percentile)', fontsize = '12')
        if mtxs_type == '16th_perc':
            ax.set_title(str(i) + ' (16th percentile)', fontsize = '12')
            
    # Remove final plot if not required
    if len(imt_list) > 3 and len(imt_list)/2 != int(len(imt_list)/2):
        ax = axs[np.unravel_index(n+1, (nrows, ncols))]
        ax.set_visible(False)
    if len(imt_list) == 1:
        axs[1].set_visible(False)
        

    pyplot.savefig(namefig, bbox_inches = 'tight', dpi = 200, pad_inches = 0.4)
    pyplot.tight_layout() 
    
    return matrix_Z


### Utils for plots

def get_cols(custom_color_flag, custom_color_list):
    """
    Get list of colors for plots
    """
    colors = ['g', 'b', 'y', 'lime', 'dodgerblue', 'gold', '0.8', '0.5', 'r',
              'm', 'mediumseagreen', 'tab:pink', 'tab:orange', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:red', 'tab:blue', 'tab:cyan',
              'tab:olive', 'aquamarine']
    
    if custom_color_flag == 'True':
        colors = custom_color_list
        
    return colors


def get_z1_z25(Z1, Z25, Vs30, region):
    """
    Get z1pt0 and z2pt5
    """
    # Set Z1 and Z25
    if  Z1 == -999:
        Z1 = _get_z1(Vs30, region)
    if  Z25 == -999:
        Z25 = _get_z25(Vs30, region)
        
    return Z1, Z25


### Trellis utils
def trellis_data(Nstd, gmpe, r_vals, mean, plus_sigma, minus_sigma, col, i, m,
                 lt_weights_gmc1, lt_vals_gmc1, lt_weights_gmc2, lt_vals_gmc2):
    """
    Plot predictions of a single GMPE (if required) and compute weighted
    predictions from logic tree(s) (again if required)
    """
    if not Nstd == 0: # If sigma is sampled from
        if 'plot_lt_only' not in str(gmpe): # If only plotting individual GMPEs
            pyplot.plot(r_vals, plus_sigma, linewidth=0.75, 
                        color=col, linestyle='-.')
            pyplot.plot(r_vals, minus_sigma, linewidth=0.75,
                        color=col, linestyle='-.')
        
        # If logic tree store values for these...
        if lt_weights_gmc1 == None:
            pass
        elif gmpe in lt_weights_gmc1:
            if lt_weights_gmc1[gmpe] != None:
                lt_vals_gmc1[gmpe] = {
                            'mean': np.exp(mean)*lt_weights_gmc1[gmpe],
                            'plus_sigma': plus_sigma*lt_weights_gmc1[gmpe],
                            'minus_sigma': minus_sigma*lt_weights_gmc1[gmpe]}
            
        if lt_weights_gmc2 == None:
            pass
        elif gmpe in lt_weights_gmc2:
            if lt_weights_gmc2[gmpe] != None:
                lt_vals_gmc2[gmpe] = {
                            'mean': np.exp(mean)*lt_weights_gmc2[gmpe],
                            'plus_sigma': plus_sigma*lt_weights_gmc2[gmpe],
                            'minus_sigma': minus_sigma*lt_weights_gmc2[gmpe]}
        
    else:                        
        if lt_weights_gmc1 == None:
            pass
        elif gmpe in lt_weights_gmc1:                                      
            if lt_weights_gmc1[gmpe] != None:
                lt_vals_gmc1[gmpe] = {
                        'mean': np.exp(mean)*lt_weights_gmc1[gmpe]}
                       
        if lt_weights_gmc2 == None:
            pass
        elif gmpe in lt_weights_gmc2:                                      
            if lt_weights_gmc2[gmpe] != None:
                lt_vals_gmc2[gmpe] = {
                        'mean': np.exp(mean)*lt_weights_gmc2[gmpe]}

    return lt_vals_gmc1, lt_vals_gmc2


def lt_trel(r_vals, Nstd, i, m,
            lt_vals_gmc1, mean_gmc1, plus_sig_gmc1, minus_sig_gmc1,
            lt_vals_gmc2, mean_gmc2, plus_sig_gmc2, minus_sig_gmc2):
    """
    If required plot spectra from the GMPE logic tree(s)
    """
    # Logic tree #1
    if lt_vals_gmc1 != {}:
        if not Nstd == 0:
               
            lt_df_gmc1 = pd.DataFrame(
                lt_vals_gmc1, index=['mean', 'plus_sigma', 'minus_sigma'])

            lt_mean_gmc1 = np.sum(lt_df_gmc1[:].loc['mean'])
            lt_plus_sigma_gmc1 = np.sum(lt_df_gmc1[:].loc['plus_sigma'])
            lt_minus_sigma_gmc1 = np.sum(lt_df_gmc1[:].loc['minus_sigma'])
   
            pyplot.plot(r_vals, lt_mean_gmc1, linewidth=2, color='k',
                        linestyle='--', label='GMC logic tree #1',
                        zorder=100)
            
            pyplot.plot(r_vals, lt_plus_sigma_gmc1, linewidth=0.75,
                        color='k', linestyle='-.', zorder=100)

            pyplot.plot(r_vals, lt_minus_sigma_gmc1, linewidth=0.75,
                        color='k', linestyle='-.', zorder=100)
            
            mean_gmc1[i,m] = lt_mean_gmc1
            plus_sig_gmc1[i,m] = lt_plus_sigma_gmc1
            minus_sig_gmc1[i,m] = lt_minus_sigma_gmc1
            
        if Nstd == 0:
            lt_df_gmc1 = pd.DataFrame(lt_vals_gmc1, index = ['mean'])
            
            lt_mean_gmc1 = np.sum(lt_df_gmc1[:].loc['mean'])
             
            pyplot.plot(r_vals, lt_mean_gmc1, linewidth=2, color='k',
                        linestyle='--', label='GMC logic tree #1')
            
            mean_gmc1[i,m] = lt_mean_gmc1
            
    # Logic tree #2
    if lt_vals_gmc2 != {}:
        if not Nstd == 0:
               
            lt_df_gmc2 = pd.DataFrame(
                lt_vals_gmc2, index=['mean', 'plus_sigma', 'minus_sigma'])
    
            lt_mean_gmc2 = np.sum(lt_df_gmc2[:].loc['mean'])
            lt_plus_sigma_gmc2 = np.sum(lt_df_gmc2[:].loc['plus_sigma'])
            lt_minus_sigma_gmc2 = np.sum(lt_df_gmc2[:].loc['minus_sigma'])
    
            pyplot.plot(r_vals, lt_mean_gmc2, linewidth=2,
                        color='tab:grey', linestyle='--',
                        label='GMC logic tree #2', zorder=100)
            
            pyplot.plot(
                r_vals, lt_plus_sigma_gmc2, linewidth=0.75,
                color='tab:grey', linestyle='-.', zorder=100)
    
            pyplot.plot(
                r_vals, lt_minus_sigma_gmc2, linewidth=0.75, 
                color='tab:grey', linestyle='-.', zorder=100)
            
            mean_gmc2[i,m] = lt_mean_gmc2
            plus_sig_gmc2[i,m] = lt_plus_sigma_gmc2
            minus_sig_gmc2[i,m] = lt_minus_sigma_gmc2
            
        if Nstd == 0:
            lt_df_gmc2 = pd.DataFrame(lt_vals_gmc2, index = ['mean'])
            
            lt_mean_gmc2 = np.sum(lt_df_gmc2[:].loc['mean'])
             
            pyplot.plot(r_vals, lt_mean_gmc2, linewidth=2,
                        color='tab:grey', linestyle='--',
                        label='GMC logic tree #2')
            
            mean_gmc2[i, m] = lt_mean_gmc2 

    return mean_gmc1, plus_sig_gmc1, minus_sig_gmc1, \
        mean_gmc2, plus_sig_gmc2, minus_sig_gmc2


def update_trellis_plots(m, i, n, l, r_vals, imt_list, dist_type):
    """
    Add titles and axis labels to trellis plots
    """
    if dist_type == 'rrup':
        label = 'Rrup (km)'
    if dist_type == 'rjb':
        label = 'Rjb (km)'
    if n == 0: #top row only
        pyplot.title('Mw = ' + str(m), fontsize='16')
    if n == len(imt_list)-1: #bottom row only
        pyplot.xlabel(label, fontsize='16') # Mod to rjb if using instead
    if l == 0: #left row only
        pyplot.ylabel(str(i) + ' (g)', fontsize='16')
    pyplot.loglog()
    pyplot.ylim(0.001, 10) # Mod if required
    #pyplot.xlim(r_vals[0], r_vals[len(r_vals)-1]) # Mod if required
                

### Spectra utils
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
    # if period extends beyond 1 s reduce interval to 0.2 s
    period = pd.Series(period)
    if max(period) > 1:
        for SA in range(0,len(period)):
            if period[SA] > 1:
                period=period.drop(SA)
        periods_to_re_add = pd.Series(np.arange(1, max_period, 0.2))
        period_df = pd.DataFrame({'periods': period, 'periods_to_re_add':
                                  periods_to_re_add, 'max_period': max_period})
        period = period_df.melt().value.dropna().unique()
    # if period extends beyond 2 s then reduce interval to 0.5 s
    period = pd.Series(period)
    if max(period) > 2:
        for SA in range(0, len(period)):
            if period[SA] > 2:
                period=period.drop(SA)
        periods_to_re_add = pd.Series(np.arange(2, max_period, 0.5))
        period_df = pd.DataFrame({'periods':period,'periods_to_re_add':
                                  periods_to_re_add,'max_period': max_period})
        period = period_df.melt().value.dropna().unique()
    # if period extends beyond 5 s then reduce interval to 1 s
    period = pd.Series(period)
    if max(period) > 5:
        for SA in range(0, len(period)):
            if period[SA] > 2:
                period=period.drop(SA)
        periods_to_re_add = pd.Series(np.arange(5, max_period, 1))
        period_df = pd.DataFrame({'periods': period, 'periods_to_re_add':
                                  periods_to_re_add, 'max_period': max_period})
        period = period_df.melt().value.dropna().unique()
    
    return period


def _get_imts(period):
    """
    Convert period floats to imt classes
    """
    # Convert from float to imt
    period = np.round(period,1)
    base_SA_string = 'SA(_)'
    imt_list = []
    for imt in range(0,len(period)):
        if imt == 0:
            SA_string = 'PGA'
        else:
            SA_string = base_SA_string.replace('_',str(period[imt]))
        imt_list.append(SA_string)
    for imt in range(0,len(imt_list)):
        imt_list[imt] = from_string(str(imt_list[imt]))
    
    return imt_list


def spectra_data(gmpe, Nstd, rs_50p, rs_plus_sigma, rs_minus_sigma,
                 lt_vals_gmc1, lt_vals_gmc2, lt_weights_gmc1, lt_weights_gmc2,
                 lt_vals_plus_sig_gmc1, lt_vals_minus_sig_gmc1,
                 lt_vals_plus_sig_gmc2, lt_vals_minus_sig_gmc2):
    """
    If required get the logic tree weighted predictions
    """
    # Logic tree #1
    if lt_weights_gmc1 == None:
        pass
    elif gmpe in lt_weights_gmc1:
        if lt_weights_gmc1[gmpe] != None:
            rs_50p_weighted_gmc1 = {}
            rs_plus_sigma_weighted_gmc1 = {}
            rs_minus_sigma_weighted_gmc1 = {}
            for idx, rs in enumerate(rs_50p):
                rs_50p_weighted_gmc1[idx] = rs_50p[idx]*lt_weights_gmc1[gmpe]
                if Nstd != 0:
                    rs_plus_sigma_weighted_gmc1[idx] = rs_plus_sigma[
                        idx]*lt_weights_gmc1[gmpe]
                    rs_minus_sigma_weighted_gmc1[idx] = rs_minus_sigma[
                        idx]*lt_weights_gmc1[gmpe]

            # If present store the weighted mean for the GMPE
            lt_vals_gmc1[gmpe] = {'mean': rs_50p_weighted_gmc1}
            
            # And if Nstd != 0 store these weighted branches too
            if Nstd != 0:
                lt_vals_plus_sig_gmc1[gmpe] = {
                    'plus_sigma': rs_plus_sigma_weighted_gmc1}
                lt_vals_minus_sig_gmc1[gmpe] = {
                    'minus_sigma': rs_minus_sigma_weighted_gmc1}
                
    # Logic tree #2              
    if lt_weights_gmc2 == None:
        pass
    elif gmpe in lt_weights_gmc2:
        if lt_weights_gmc2[gmpe] != None:
            rs_50p_weighted_gmc2 = {}
            rs_plus_sigma_weighted_gmc2, rs_minus_sigma_weighted_gmc2 = {}, {}
            for idx, rs in enumerate(rs_50p):
                rs_50p_weighted_gmc2[idx] = rs_50p[idx]*lt_weights_gmc2[gmpe]
                if Nstd != 0:
                    rs_plus_sigma_weighted_gmc2[idx] = rs_plus_sigma[
                        idx]*lt_weights_gmc2[gmpe]
                    rs_minus_sigma_weighted_gmc2[idx] = rs_minus_sigma[
                        idx]*lt_weights_gmc2[gmpe]
                
            # If present store the weighted mean for the GMPE
            lt_vals_gmc2[gmpe] = {'mean': rs_50p_weighted_gmc2}
            
            # And if Nstd != 0 store these weighted branches too
            if Nstd != 0:
                lt_vals_plus_sig_gmc2[gmpe] = {
                    'plus_sigma': rs_plus_sigma_weighted_gmc2}
                lt_vals_minus_sig_gmc2[gmpe] = {
                    'minus_sigma': rs_minus_sigma_weighted_gmc2}
        
    return lt_vals_gmc1, lt_vals_plus_sig_gmc1, lt_vals_minus_sig_gmc1, \
           lt_vals_gmc2, lt_vals_plus_sig_gmc2, lt_vals_minus_sig_gmc2


def lt_spectra(ax1, gmpe, gmpe_list, Nstd, period, gmc1_or_gmc2,
               lt_vals_gmc, lt_vals_plus_sig_gmc, lt_vals_minus_sig_gmc):
    """
    If required plot spectra from the GMPE logic tree(s)
    """    
    if gmc1_or_gmc2 == 'gmc1':
        check = 'lt_weight_gmc1'
        label = 'Logic Tree 1'
        col = 'k'
    if gmc1_or_gmc2 == 'gmc2':
        check = 'lt_weight_gmc2'
        label = 'Logic Tree 2'
        col = 'tab:grey'
    
    # Plot
    if lt_vals_gmc != {}:
        lt_df_gmc = pd.DataFrame(lt_vals_gmc, index=['mean'])
        if Nstd != 0:
            lt_df_plus_sigma_gmc = pd.DataFrame(lt_vals_plus_sig_gmc,
                                                 index=['plus_sigma'])
            lt_df_minus_sigma_gmc = pd.DataFrame(lt_vals_minus_sig_gmc,
                                                  index=['minus_sigma'])
            
        wt_mean_per_gmpe_gmc = {}
        wt_plus_sigma_per_gmpe_gmc, wt_minus_sigma_per_gmpe_gmc = {}, {}
        
        for gmpe in gmpe_list:
            if check in str(gmpe):
                wt_mean_per_gmpe_gmc[gmpe] = np.array(pd.Series(
                    lt_df_gmc[gmpe].loc['mean']))
                if Nstd != 0:
                    wt_plus_sigma_per_gmpe_gmc[gmpe] = np.array(
                        pd.Series(lt_df_plus_sigma_gmc[gmpe].loc[
                            'plus_sigma']))
                    wt_minus_sigma_per_gmpe_gmc[gmpe] = np.array(
                        pd.Series(lt_df_minus_sigma_gmc[gmpe].loc[
                            'minus_sigma']))
            
        lt_df_gmc = pd.DataFrame(wt_mean_per_gmpe_gmc, index=period)
        lt_df_plus_sigma_gmc = pd.DataFrame(wt_plus_sigma_per_gmpe_gmc,
                                             index=period)
        lt_df_minus_sigma_gmc = pd.DataFrame(wt_minus_sigma_per_gmpe_gmc,
                                              index=period)
        
        lt_mean_per_period_gmc = {}
        lt_plus_sigma_per_period_gmc, lt_minus_sigma_per_period_gmc = {}, {}
        for idx, imt in enumerate(period):
            lt_mean_per_period_gmc[imt] = np.sum(lt_df_gmc.loc[imt])
            if Nstd != 0:
                lt_plus_sigma_per_period_gmc[imt] = np.sum(
                    lt_df_plus_sigma_gmc.loc[imt])
                lt_minus_sigma_per_period_gmc[imt] = np.sum(
                    lt_df_minus_sigma_gmc.loc[imt])
        
        # Plot logic tree #
        ax1.plot(period, np.array(pd.Series(lt_mean_per_period_gmc)), linewidth=2,
                 color=col, linestyle='--', label = label, zorder=100)
        
        # Plot mean plus sigma and mean minus sigma if required
        if Nstd != 0:
            ax1.plot(period, np.array(pd.Series(lt_plus_sigma_per_period_gmc)),
                     linewidth=0.75, color=col, linestyle='-.', zorder=100)
            
            ax1.plot(period, np.array(pd.Series(lt_minus_sigma_per_period_gmc)),
                     linewidth=0.75, color=col, linestyle='-.', zorder=100)
            

def load_obs_spectra(obs_spectra):
    """
    If an obs spectra has been specified get values from csvs for comparison
    of observed spectra and spectra computed using GMPE predictions
    """
    # Get values from obs_spectra dataframe...
    eq_id = str(obs_spectra['EQ ID'].iloc[0])
    mw = float(obs_spectra['Mw'].iloc[0])
    dep = float(obs_spectra['Depth (km)'].iloc[0])
    rrup = float(obs_spectra['Rrup (km)'].iloc[0])
    st_id = str(obs_spectra['Station Code'].iloc[0])
    mag_list = np.array([mw])
    dist_list = np.array([rrup])
    depth = np.array([dep])
    strike = float(obs_spectra['Strike'].iloc[0])
    dip = float(obs_spectra['Dip'].iloc[0])
    rake = float(obs_spectra['Rake'].iloc[0])
    vs30 = float(obs_spectra['Vs30'].iloc[0])
    ztor = [str(obs_spectra['ztor'].iloc[0])] # Must be list for iterating
    if ztor == 'None':
        ztor = None
    trt = str(obs_spectra['trt'].iloc[0])
    up_or_down_dip = float(
        obs_spectra['Site up-dip of rupture (1 = True, 0 = False)'].iloc[0])
    
    return eq_id, mw, dep, rrup, st_id, mag_list, dist_list, depth, strike,
    dip, rake, vs30, ztor, trt, up_or_down_dip


def plot_obs_spectra(ax1, obs_spectra, g, gmpe_list,  dep=None, rrup=None,
                     mw=None, eq_id=None, st_id=None):
    """
    Check if an observed spectra must be plotted, and if so plot
    """
    # Plot an observed spectra if inputted...
    if obs_spectra is not None and g == len(gmpe_list)-1:
        # Get label for spectra plot
        obs_string = (eq_id + '\nrecorded at ' + st_id + ' (Rrup = '
                      + str(rrup) + ' km, ' + '\nMw = ' + str(mw) +
                      ', depth = ' + str(dep) + ' km)')
        # Plot the observed spectra
        ax1.plot(obs_spectra['Period (s)'], obs_spectra['SA (g)'],
                 color='r', linewidth=3, linestyle='-',
                 label=obs_string)    
        
        
def update_spec_plots(ax1, ax2, m, i, n, l, dist_list):
    """
    Add titles and axis labels to spectra plots
    """
    ax1.set_title('Mw = ' + str(m) + ', R = ' + str(i) + ' km',
                  fontsize=16, y=1.0, pad=-16)
    ax2.set_title('Mw = ' + str(m) + ', R = ' + str(i) + ' km',
                  fontsize=16, y=1.0, pad=-16)
    if n == len(dist_list)-1: #bottom row only
        ax1.set_xlabel('Period (s)', fontsize=16)
        ax2.set_xlabel('Period (s)', fontsize=16)
    if l == 0: # left row only
        ax1.set_ylabel('Sa (g)', fontsize=16) 
        ax2.set_ylabel(r'$\sigma$', fontsize=16) 


def save_spectra_plot(f1, f2, obs_spectra, output_dir, eq_id=None, st_id=None):
    """
    Save the plotted response spectra
    """
    if obs_spectra is None:
        f1.savefig(os.path.join(output_dir, 'ResponseSpectra.png'),
                     bbox_inches = 'tight', dpi = 200, pad_inches = 0.2)
    else:
        rec_str = str(eq_id + '_recorded_at_' + st_id)
        rec_str = rec_str.replace(' ','_').replace(':','_').replace('-','_')
        fname = 'ResponseSpectra_' + rec_str + '.png'
        f1.savefig(os.path.join(output_dir, fname), bbox_inches = 'tight',
                   dpi = 200, pad_inches = 0.2)
        
    # Save sigma plot
    f2.savefig(os.path.join(output_dir,'sigma.png'), bbox_inches = 'tight',
               dpi = 200, pad_inches = 0.2)