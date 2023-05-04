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

def plot_trellis_util(rake, strike, dip, depth, Z1, Z25, Vs30, region,
                 imt_list, mag_list, maxR, gmpe_list, aratio, Nstd,
                 output_directory, custom_color_flag, custom_color_list,
                 eshm20_region, lt_weights = None):
    """
    Generate trellis plots for given run configuration
    """
    # Plots: color for GMPEs
    colors=['r', 'g', 'b', 'y','lime','k','dodgerblue', 'gold', '0.8',
            'mediumseagreen', '0.5','tab:orange', 'tab:purple', 'tab:brown',
            'tab:pink', 'tab:grey', 'tab:cyan', 'tab:olive', 'tab:red',
            'aquamarine']
    if custom_color_flag == 'True':
        colors = custom_color_list
            
    step = 1
    
    # Set Z1 and Z25
    if  Z1 == -999:
        Z1 = _get_z1(Vs30,region)

    if  Z25 == -999:
        Z25 = _get_z25(Vs30,region)
    
    fig = pyplot.figure(figsize=(len(mag_list)*5, len(imt_list)*4))
    
    store_trellis_values = {}
    lt_mean_store = {}
    lt_plus_sigma_store = {}
    lt_minus_sigma_store = {}
    for n, i in enumerate(imt_list): #iterate though imt_list

        for l, m in enumerate(mag_list):  #iterate though mag_list

            fig.add_subplot(len(imt_list), len(mag_list), l+1+n*len(
                mag_list)) #(#vert, #hor, #subplot)
            
            store_lt_branch_values = {}
            
            for g, gmpe in enumerate(gmpe_list): 
                
                strike_g, dip_g, depth_g, aratio_g = _param_gmpes(
                    gmpe, strike, dip, depth[l], aratio, rake) 

                gmm = valid.gsim(gmpe)
                col=colors[g]

                if not Nstd ==0:
                    gmm, gmpe_sigma_flag = mgmpe_check(gmpe, str(i), 
                                                       task = 'comparison')
                else:
                    pass

                mean, std, distances = att_curves(gmm,depth[l],m,aratio_g,
                                                 strike_g,dip_g,rake,Vs30,
                                                 Z1,Z25,maxR,step,i,1,
                                                 eshm20_region) 
                mean = mean[0][0]
                std = std[0][0]
                
                if 'lt_weight_plot_lt_only' not in str(gmpe):
                    pyplot.plot(distances, np.exp(mean), color=col,
                                linewidth=2, linestyle='-', label=gmpe)
                else:
                    pass
                
                plus_sigma = np.exp(mean+Nstd*std[0])
                minus_sigma = np.exp(mean-Nstd*std[0])
                
                # Plot Sigma                
                if not Nstd==0:
                    if 'lt_weight_plot_lt_only' not in str(gmpe):
                        pyplot.plot(distances, plus_sigma, linewidth=0.75,
                                    color=col, linestyle='-.')
                        pyplot.plot(distances, minus_sigma, linewidth=0.75,
                                    color=col, linestyle='-.')
                    else:
                        pass
                                     
                    store_trellis_values['IM = ' + str(i), 'Magnitude = ' 
                                         + str(m), str(gmpe).replace(
                                             '\n', ', ').replace('[', '').replace(
                                                 ']', '')] = [np.array(np.exp(mean)),
                                                           np.array(plus_sigma),
                                                           np.array(minus_sigma),
                                                           np.array(distances)]
                    
                    if lt_weights == None:
                        pass
                    elif gmpe in lt_weights:
                        if lt_weights[gmpe] != None:
                                store_lt_branch_values[gmpe] = {
                                        'mean': np.exp(mean)*lt_weights[gmpe],
                                        'plus_sigma': plus_sigma*lt_weights[gmpe],
                                        'minus_sigma': minus_sigma*lt_weights[gmpe]}
                        else:
                            pass 
                    
                else:
                    store_trellis_values['IM = ' + str(i), 'Magnitude = ' +
                                             str(m), str(gmpe).replace(
                                                 '\n', ', ').replace(
                                                     '[', '').replace(']', '')
                                                     ] = [np.array(np.exp(mean)),
                                                              np.array(distances)]
                                                          
                    if lt_weights == None:
                        pass
                    elif gmpe in lt_weights:                                      
                        if lt_weights[gmpe] != None:
                            store_lt_branch_values[gmpe] = {
                                    'mean': np.exp(mean)*lt_weights[gmpe]}
                        else:
                            pass
                                                              
                if n == 0: #top row only
                    pyplot.title('Mw=' + str(m), fontsize='16')
                if n == len(imt_list)-1: #bottom row only
                    pyplot.xlabel('Rrup (km)', fontsize='14')
                if l == 0: #left row only
                    pyplot.ylabel(str(i) + ' (g)', fontsize='16')

                pyplot.loglog()
                pyplot.ylim(0.001, 10)
                pyplot.xlim(1, maxR)
                
            pyplot.grid(axis='both', which='both', alpha=0.5)
        
            # Plot logic tree for the IMT-mag combination if weights specified
            logic_tree_config = 'Inputted GMPE logic tree config.'
            
            if store_lt_branch_values != {}:
                if not Nstd == 0:
                       
                    lt_df = pd.DataFrame(store_lt_branch_values,
                                         index = ['mean', 'plus_sigma',
                                                  'minus_sigma'])

                    lt_mean = np.sum(lt_df[:].loc['mean'])
                    lt_plus_sigma = np.sum(lt_df[:].loc['plus_sigma'])
                    lt_minus_sigma = np.sum(lt_df[:].loc['minus_sigma'])
           
                    pyplot.plot(distances, lt_mean, linewidth = 2, color = 'm',
                                linestyle = '-', label = logic_tree_config, zorder = 100)
                    
                    pyplot.plot(distances, lt_plus_sigma, linewidth = 0.75,
                                color = 'm', linestyle = '-.', zorder = 100)
        
                    pyplot.plot(distances, lt_minus_sigma, linewidth = 0.75,
                                color = 'm', linestyle = '-.', zorder = 100)
                    
                    lt_mean_store[i,m] = lt_mean
                    lt_plus_sigma_store[i,m] = lt_plus_sigma
                    lt_minus_sigma_store[i,m] = lt_minus_sigma
                    
                if Nstd == 0:
                    lt_df = pd.DataFrame(store_lt_branch_values, index = ['mean'])
                    
                    lt_mean = np.sum(lt_df[:].loc['mean'])
                     
                    pyplot.plot(distances, lt_mean, linewidth = 2, color = 'm',
                                linestyle = '-', label = logic_tree_config)
                    
                    lt_mean_store[i,m] = lt_mean

    pyplot.legend(loc="center left", bbox_to_anchor=(1.1, 1.05), fontsize='16')
    pyplot.savefig(os.path.join(output_directory,'TrellisPlots.png'),
                   bbox_inches='tight',dpi=200,pad_inches = 0.2)
    pyplot.show()
    pyplot.tight_layout()    

    # Export values to csv
    if not Nstd == 0:
        trellis_value_df = pd.DataFrame(store_trellis_values,
                                        index = ['Mean (g)',
                                                 'Plus %s sigma (g)' %Nstd,
                                                 'Minus %s sigma (g)' %Nstd,
                                                     'Distance (km)'])
        if lt_weights != None:
            for n, i in enumerate(imt_list): #iterate though imt_list
                for l, m in enumerate(mag_list):  #iterate through mag_list
                    trellis_value_df['IM = ' + str(i),
                                     'Magnitude = ' + str(m),'GMPE logic tree'] = [
                                         lt_mean_store[i,m],lt_plus_sigma_store[i,m],
                                         lt_minus_sigma_store[i,m], distances]
        else:
            pass
                                         
    if Nstd == 0:
        trellis_value_df = pd.DataFrame(store_trellis_values,
                                        index = ['Mean (g)', 'Distance (km)'])
        if lt_weights != None:
            for n, i in enumerate(imt_list): #iterate though imt_list
                for l, m in enumerate(mag_list):  #iterate through mag_list
                    trellis_value_df['IM = ' + str(i), 'Magnitude = ' + str(m),
                                     'GMPE logic tree'] = [lt_mean_store[i,m],
                                                          distances]                                                         
        else:
            pass
    display(trellis_value_df)
    trellis_value_df.to_csv(os.path.join(output_directory, 'trellis_values.csv'))
    
    
def plot_spectra_util(rake, strike, dip, depth, Z1, Z25, Vs30, region,
                      max_period, mag_list, dist_list, gmpe_list, aratio, Nstd,
                      output_directory, custom_color_flag, custom_color_list,
                      eshm20_region, lt_weights = None):
    """
    Plot response spectra and sigma w.r.t. spectral period for given run
    configuration
    :param dist_list:
        Array of distances to generate response spectra and sigma plots for 
    :param max_period:
        Maximum period to compute plots for (note an error will be returned if
        this exceeds the maximum spectral period of a GMPE listed in gmpe_list)
    """
    # Set initial periods with constant spacing of 0.1
    period = list(np.round(np.arange(0,max_period,0.1),1))
    period.append(max_period)
    # if period extends beyond 1 s reduce interval to 0.2 s
    period = pd.Series(period)
    if max(period) > 1:
        for SA in range(0,len(period)):
            if period[SA] > 1:
                period=period.drop(SA)
        periods_to_re_add = pd.Series(np.arange(1,max_period,0.2))
        period_df = pd.DataFrame({'periods':period,'periods_to_re_add':
                                  periods_to_re_add, 'max_period': max_period})
        period = period_df.melt().value.dropna().unique()
    # if period extends beyond 2 s then reduce interval to 0.5 s
    period = pd.Series(period)
    if max(period) > 2:
        for SA in range(0,len(period)):
            if period[SA] > 2:
                period=period.drop(SA)
        periods_to_re_add = pd.Series(np.arange(2,max_period,0.5))
        period_df = pd.DataFrame({'periods':period,'periods_to_re_add':
                                  periods_to_re_add,'max_period': max_period})
        period = period_df.melt().value.dropna().unique()
    # if period extends beyond 5 s then reduce interval to 1 s
    period = pd.Series(period)
    if max(period) > 5:
        for SA in range(0,len(period)):
            if period[SA] > 2:
                period=period.drop(SA)
        periods_to_re_add = pd.Series(np.arange(5,max_period,1))
        period_df = pd.DataFrame({'periods':period,'periods_to_re_add':
                                  periods_to_re_add,'max_period': max_period})
        period = period_df.melt().value.dropna().unique()
        
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
    
    # Set Z1 and Z25
    if  Z1 == -999:
        Z1 = _get_z1(Vs30,region)

    if  Z25 == -999:
        Z25 = _get_z25(Vs30,region)
        
    # Plots: color for GMPEs
    colors=['r', 'g', 'b', 'y','lime','k','dodgerblue','gold','0.8',
            'mediumseagreen','0.5','tab:orange', 'tab:purple','tab:brown',
            'tab:pink', 'tab:grey', 'tab:cyan', 'tab:olive', 'tab:red',
            'aquamarine']
    if custom_color_flag == 'True':
        colors = custom_color_list
    
    fig1 = pyplot.figure(figsize=(len(mag_list)*5, len(dist_list)*4))
    pyplot.rcParams.update({'font.size': 16})# response spectra
    fig2 = pyplot.figure(figsize=(len(mag_list)*5, len(dist_list)*4))
    pyplot.rcParams.update({'font.size': 16})# sigma
    
    # Set dicts to store values
    store_spectra_values = {}
    store_lt_branch_values = {}
    store_lt_mean_per_dist_mag = {}
    
    store_lt_branch_values_plus_sigma = OrderedDict([(gmm,
                                  {}) for gmm in gmpe_list])    
    store_lt_branch_values_minus_sigma = OrderedDict([(gmm,
                                  {}) for gmm in gmpe_list])
    
    store_lt_plus_sigma_per_dist_mag = {}
    store_lt_minus_sigma_per_dist_mag = {}

    for n, i in enumerate(dist_list): #iterate though dist_list
        
        for l, m in enumerate(mag_list):  #iterate through mag_list
            
            ax1 = fig1.add_subplot(len(dist_list), len(mag_list), l+1+n*len(
                mag_list)) #(#vert, #hor, #subplot)
            ax2 = fig2.add_subplot(len(dist_list), len(mag_list), l+1+n*len(
                mag_list)) #(#vert, #hor, #subplot)

            for g, gmpe in enumerate(gmpe_list): 
                
                col=colors[g]
                gmm = valid.gsim(gmpe)
                
                strike_g, dip_g, depth_g, aratio_g = _param_gmpes(gmpe, strike,
                                                                  dip, 
                                                                  depth[l],
                                                                  aratio, rake)
                
                rs_50p, rs_plus_sigma, rs_minus_sigma, sigma = [], [], [], []
                
                for k, imt in enumerate(imt_list): 
                    mu, std, distances = att_curves(gmm,depth[l],m,aratio_g,
                                                    strike_g,dip_g,rake,Vs30,
                                                    Z1,Z25,300,0.1,imt,1,
                                                    eshm20_region) 
                    
                    mu = mu[0][0]
                    f = interpolate.interp1d(distances,mu)
                    rs_50p_dist = np.exp(f(i))
                    
                    f1 = interpolate.interp1d(distances,std[0])
                    sigma_dist = f1(i)
                    
                    if Nstd != 0:
                            rs_plus_sigma_dist = np.exp(f(i)+(Nstd*sigma_dist))
                            rs_minus_sigma_dist = np.exp(f(i)-(Nstd*sigma_dist))
                    else:
                        pass
                 
                    rs_50p.append(rs_50p_dist)
                    if Nstd != 0:
                        rs_plus_sigma.append(rs_plus_sigma_dist)
                        rs_minus_sigma.append(rs_minus_sigma_dist)
                    sigma.append(sigma_dist)
                    
                if 'lt_weight_plot_lt_only' not in str(gmpe):
                    ax1.plot(period, rs_50p, color=col, linewidth=2, linestyle='-',
                             label=gmpe)
                    ax2.plot(period, sigma, color=col, linewidth=2, linestyle='-',
                             label=gmpe)
                    if Nstd != 0:
                        ax1.plot(period, rs_plus_sigma, color=col, linewidth=0.75,
                                 linestyle='-.')
                        ax1.plot(period, rs_minus_sigma, color=col, linewidth=0.75,
                                 linestyle='-.')
                else:
                    pass
                
                sigma_store = []
                for idx_sigma, value_sigma in enumerate(rs_plus_sigma):       
                    sigma_store.append(value_sigma[0])
                    
                if Nstd != 0:
                    plus_sigma_store = []
                    minus_sigma_store = []
                    for idx_50p_plus_sigma, value_50p_plus_sigma in enumerate(
                            rs_plus_sigma):
                        plus_sigma_store.append(value_50p_plus_sigma[0])
                    for idx_50p_minus_sigma, value_50p_minus_sigma in enumerate(
                            rs_minus_sigma):
                        minus_sigma_store.append(value_50p_minus_sigma[0])

                    store_spectra_values['Distance = %s km' %i, 'Magnitude = '
                                         + str(m), str(gmpe).replace(
                                             '\n', ', ').replace('[', '').replace(
                                                 ']', '')] = [np.array(period),
                                                            np.array(rs_50p),
                                                            plus_sigma_store,
                                                            minus_sigma_store,
                                                            sigma_store]                  
                else:
                    store_spectra_values['Distance = %s km' %i, 'Magnitude = '
                                         + str(m), str(gmpe).replace(
                                             '\n', ', ').replace('[', '').replace(
                                                 ']', '')] = [np.array(period),
                                                            np.array(rs_50p),
                                                            sigma_store]
                # Check if weight provided for the GMPE                            
                if lt_weights == None:
                    pass
                elif gmpe in lt_weights:
                    if lt_weights[gmpe] != None:
                        rs_50p_weighted = {}
                        rs_plus_sigma_weighted = {}
                        rs_minus_sigma_weighted = {}
                        for idx, rs in enumerate(rs_50p):
                            rs_50p_weighted[idx] = rs_50p[idx]*lt_weights[gmpe]
                            if Nstd != 0:
                                rs_plus_sigma_weighted[idx] = rs_plus_sigma[
                                    idx]*lt_weights[gmpe]
                                rs_minus_sigma_weighted[idx] = rs_minus_sigma[
                                    idx]*lt_weights[gmpe]
                            else:
                                pass
                            
                        # If present store the weighted mean for the GMPE
                        store_lt_branch_values[gmpe] = {'mean': rs_50p_weighted}
                        
                        # And if Nstd != 0 store these weighted branches too
                        if Nstd != 0:
                            store_lt_branch_values_plus_sigma[gmpe] = {
                                'plus_sigma': rs_plus_sigma_weighted}
                            store_lt_branch_values_minus_sigma[gmpe] = {
                                'minus_sigma': rs_minus_sigma_weighted}
                            
                            #print(gmpe, store_lt_branch_values_plus_sigma)
                            
                # Continue with plot creation
                ax1.set_title('Mw = ' + str(m) + ' - R = ' + str(i) + ' km',
                              fontsize=16, y=1.0, pad=-16)
                ax2.set_title('Mw = ' + str(m) + ' - R = ' + str(i) + ' km',
                              fontsize=16, y=1.0, pad=-16)
                if n == len(dist_list)-1: #bottom row only
                    ax1.set_xlabel('Period (s)', fontsize=16)
                    ax2.set_xlabel('Period (s)', fontsize=16)
                if l == 0: # left row only
                    ax1.set_ylabel('Sa (g)', fontsize=16) 
                    ax2.set_ylabel(r'$\sigma$', fontsize=16) 
            ax1.grid(True)
            ax2.grid(True)
            ax2.set_ylim(0.3, 1)
            
            # Plot logic tree for the dist-mag combination if weights specified
            logic_tree_config = 'Inputted GMPE logic tree config.'
            
            # Create the dataframe of stored values for mean etc
            if store_lt_branch_values != {}:
                lt_df = pd.DataFrame(store_lt_branch_values, index = ['mean'])
                if Nstd != 0:
                    lt_df_plus_sigma = pd.DataFrame(
                        store_lt_branch_values_plus_sigma, index = ['plus_sigma'])
                    lt_df_minus_sigma = pd.DataFrame(
                        store_lt_branch_values_minus_sigma, index = ['minus_sigma'])
                    
                weighted_mean_per_gmpe = {}
                weighted_plus_sigma_per_gmpe = {}
                weighted_minus_sigma_per_gmpe = {}
                
                for gmpe in gmpe_list:
                    if 'lt_weight' in str(gmpe):
                        weighted_mean_per_gmpe[gmpe] = np.array(pd.Series(lt_df[
                        gmpe].loc['mean']))
                        if Nstd != 0:
                            weighted_plus_sigma_per_gmpe[gmpe] = np.array(
                                pd.Series(lt_df_plus_sigma[gmpe].loc[
                                    'plus_sigma']))
                            weighted_minus_sigma_per_gmpe[gmpe] = np.array(
                                pd.Series(lt_df_minus_sigma[gmpe].loc[
                                    'minus_sigma']))
                    else:
                        pass
                    
                lt_df = pd.DataFrame(weighted_mean_per_gmpe, index = period)
                lt_df_plus_sigma = pd.DataFrame(weighted_plus_sigma_per_gmpe,
                                                index = period)
                lt_df_minus_sigma = pd.DataFrame(weighted_minus_sigma_per_gmpe,
                                                 index = period)
                lt_mean_per_period = {}
                lt_plus_sigma_per_period = {}
                lt_minus_sigma_per_period = {}
                for idx, imt in enumerate(period):
                    lt_mean_per_period[imt] = np.sum(lt_df.loc[imt])
                    if Nstd != 0:
                        lt_plus_sigma_per_period[imt] = np.sum(
                            lt_df_plus_sigma.loc[imt])
                        lt_minus_sigma_per_period[imt] = np.sum(
                            lt_df_minus_sigma.loc[imt])
                
                # Plot the logic tree
                ax1.plot(period, np.array(pd.Series(lt_mean_per_period)),
                         linewidth = 2, color = 'm', linestyle = '-',
                         label = logic_tree_config, zorder = 100)
                
                # Plot mean plus sigma and mean minus sigma if required
                if Nstd != 0:
                    ax1.plot(period, np.array(pd.Series(lt_plus_sigma_per_period)),
                             linewidth = 0.75, color = 'm', linestyle = '-.',
                             zorder = 100)
                    
                    ax1.plot(period, np.array(pd.Series(lt_minus_sigma_per_period)),
                             linewidth = 0.75, color = 'm', linestyle = '-.',
                             zorder = 100)
                else:
                    pass
                
                # Store the logic tree plot data for .csv output
                store_lt_mean_per_dist_mag[i,m] = lt_mean_per_period
                if Nstd != 0:
                    store_lt_plus_sigma_per_dist_mag[i,m] = lt_plus_sigma_per_period
                    store_lt_minus_sigma_per_dist_mag[i,m] = lt_minus_sigma_per_period
                
    # Finalise the plots
    ax1.legend(loc="center left", bbox_to_anchor=(1.1, 1.05), fontsize='16')
    ax2.legend(loc="center left", bbox_to_anchor=(1.1, 1.05), fontsize='16')
    fig2.savefig(os.path.join(output_directory,'sigma.png'),
                 bbox_inches='tight',dpi=200,pad_inches = 0.2)
    fig1.savefig(os.path.join(output_directory,'ResponseSpectra.png'),
                 bbox_inches='tight',dpi=200,pad_inches = 0.2)
    
    # Export values to csv
    if Nstd != 0:
        spectra_value_df = pd.DataFrame(store_spectra_values,
                                        index = ['Periods', 'Median (g)',
                                                 'Plus %s sigma (g)' %Nstd,
                                                 'Minus %s sigma (g)' %Nstd,
                                                 'GMPE Sigma (natural log)'])
    else: 
        spectra_value_df = pd.DataFrame(store_spectra_values,
                                        index = ['Periods', 'Median (g)',
                                                 'GMPE Sigma (natural log)'])
        
    if lt_weights != None:
        store_plus_sigma_per_dist_mag = []
        store_minus_sigma_per_dist_mag = []
        for n, i in enumerate(dist_list): #iterate though dist_list
            for l, m in enumerate(mag_list):  #iterate through mag_list
                if Nstd != 0:
                    dict_keys = store_lt_plus_sigma_per_dist_mag[i,m].keys()
                    for key in dict_keys:
                        store_plus_sigma_per_dist_mag.append(
                            store_lt_plus_sigma_per_dist_mag[i,m][key])
                        store_minus_sigma_per_dist_mag.append(
                            store_lt_minus_sigma_per_dist_mag[i,m][key])
                    spectra_value_df[
                        'Distance = ' + str(i) + 'km', 'Magnitude = ' + str(m),
                        'GMPE logic tree'] = [np.array(period), np.array(pd.Series(
                            store_lt_mean_per_dist_mag[i,m])),
                            store_plus_sigma_per_dist_mag,
                            store_minus_sigma_per_dist_mag
                            , '-']
                else:
                    spectra_value_df[
                        'Distance = ' + str(i) + 'km', 'Magnitude = ' + str(m),
                        'GMPE logic tree'] = [np.array(period), np.array(pd.Series(
                            store_lt_mean_per_dist_mag[i,m])), '-']         
    else:
        pass
    display(spectra_value_df)
    spectra_value_df.to_csv(os.path.join(output_directory, 'spectra_values.csv'))


def compute_matrix_gmpes(imt_list, mag_list, gmpe_list, rake, strike,
                         dip, depth, Z1, Z25, Vs30, region,  maxR,  aratio,
                         eshm20_region,mtxs_type):
    """
    Compute matrix of median ground-motion predictions for each gmpe for the
    given run configuration for use within Euclidean distance matrix plots,
    Sammons Mapping and hierarchical clustering plots
    :param mtxs_type:
        type of predicted ground-motion matrix being computed in
        compute_matrix_gmpes (either median, 84th or 16th percentile)
    """
    step = 1
    if  Z1 == -999:
        Z1 = _get_z1(Vs30,region)

    if  Z25 == -999:
        Z25 = _get_z25(Vs30,region)
    
    # define the matrix with the spectral accelerations
    mtxs_median = {}
 
    for n, i in enumerate(imt_list): #iterate though imt_list

        matrix_medians=np.zeros((len(gmpe_list),(len(mag_list)*int((
            maxR/step)))))

        for g, gmpe in enumerate(gmpe_list): 
            medians = []
            sigmas = []
            for l, m in enumerate(mag_list): #iterate though mag_list
        
                strike_g, dip_g, depth_g, aratio_g = _param_gmpes(
                    gmpe,strike, dip, depth[l], aratio, rake) 

                gmm, gmpe_sigma_flag = mgmpe_check(gmpe, str(i),
                                                           task = 'comparison')

                mean, std, distances = att_curves(gmm,depth[l],m,aratio_g,
                                                  strike_g,dip_g,rake,Vs30,Z1,
                                                  Z25,maxR,step,i,1,eshm20_region) 
                
                if mtxs_type == 'median':
                    medians = np.append(medians,(np.exp(mean)))
                if mtxs_type == '84th_perc':
                    Nstd = 1 # median + 1std = ~84th percentile
                    medians = np.append(medians,(np.exp(mean+Nstd*std[0])))
                if mtxs_type == '16th_perc':
                    Nstd = 1 # median - 1std = ~16th percentile
                    medians = np.append(medians,(np.exp(mean-Nstd*std[0])))   
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
    # Set the axis and title
        # Plot
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

    pyplot.savefig(namefig, bbox_inches='tight',dpi=200,pad_inches = 0.2)
    pyplot.show()
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
    # Plots: color for GMPEs
    colors=['r', 'g', 'b', 'y','lime','k','dodgerblue','gold','0.8',
            'mediumseagreen','0.5','tab:orange', 'tab:purple','tab:brown',
            'tab:pink', 'tab:grey', 'tab:cyan', 'tab:olive', 'tab:red',
            'aquamarine']
    if custom_color_flag == 'True':
        colors = custom_color_list
            
    texts = []
    
    # Rows and cols  
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
        coo, cost = sammon(data ,display=1)
        
        fig.add_subplot(nrows, ncols, n+1) #(#vert, #hor, #subplot)

        for g, gmpe in enumerate(gmpe_list): 
            col=colors[g]
            pyplot.plot(coo[g,0], coo[g,1], 'o', markersize=9, color=colors[
                g], label=gmpe)
            texts.append(pyplot.text(coo[g, 0]+np.abs(coo[g, 0])*0.02,
                                     coo[g, 1]+np.abs(coo[g, 1])*0.,
                                     gmpe_list[g], ha='left', color=col))

        pyplot.title(str(i), fontsize='16')
        if mtxs_type == 'median':
            pyplot.title(str(i) + ' (median)', fontsize = '14')
        if mtxs_type == '84th_perc':
            pyplot.title(str(i) + ' (84th percentile)', fontsize = '14')
        if mtxs_type == '16th_perc':
            pyplot.title(str(i) + ' (16th percentile)', fontsize = '14')
        pyplot.grid(axis='both', which='both', alpha=0.5)

    pyplot.legend(loc="center left", bbox_to_anchor=(1.25, 0.50), fontsize='16')
    pyplot.savefig(namefig, bbox_inches='tight',dpi=200,pad_inches = 0.2)
    pyplot.show()
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
    ncols = 2
    
    if len(imt_list) < 3:
        nrows = 1
    else:
        nrows = int(np.ceil(len(imt_list) / 2)) 
    
    # Clustering Plots
    matrix_Z = {}
    ymax =  [0] * len(imt_list)

    # Loop over IMTs
    for n, i in enumerate(imt_list):

        # Get the data matrix
        data = mtxs[n]

        # Agglomerative clustering
        Z = hierarchy.linkage(data, method='ward', metric='euclidean',
                              optimal_ordering=True)
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

    pyplot.savefig(namefig, bbox_inches='tight',dpi=200,pad_inches = 0.4)
    pyplot.show()
    pyplot.tight_layout() 
    return matrix_Z