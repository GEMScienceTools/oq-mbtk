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
import os
from matplotlib import pyplot
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

from openquake.smt.comparison.sammons import sammon
from openquake.hazardlib import valid
from openquake.smt.comparison.utils_gmpes import att_curves, _get_z1, _get_z25, _param_gmpes

def plot_trellis_util(rake, strike, dip, depth, Z1, Z25, Vs30, region,
                 imt_list, mag_list, maxR, gmpe_list, aratio, Nstd, name,
                 output_directory):
    """
    Generate trellis plots for given run configuration
    """
    # Plots: color for GMPEs
    colors=['r', 'g', 'b', 'y','lime','dodgerblue', 'k', 'gold']
    step = 1
    # Npoints=maxR/step
    threshold = 0.01
    
    if  Z1 == -999:
        Z1 = _get_z1(Vs30,region)

    if  Z25 == -999:
        Z25 = _get_z25(Vs30,region)
    
    fig = pyplot.figure(figsize=(len(mag_list)*5, len(imt_list)*4))

    for n, i in enumerate(imt_list): #iterate though imt_list

        for l, m in enumerate(mag_list):  #iterate though mag_list

            fig.add_subplot(len(imt_list), len(mag_list), l+1+n*len(
                mag_list)) #(#vert, #hor, #subplot)

            for g, gmpe in enumerate(gmpe_list): 
                
                strike_g, dip_g, depth_g, aratio_g = _param_gmpes(
                    gmpe, strike, dip, depth[l], aratio, rake) 

                gmm = valid.gsim(gmpe)
                col=colors[g]

                mean, std, distances = att_curves(gmm,depth[l],m,aratio_g,
                                                 strike_g,dip_g,rake,Vs30,
                                                 Z1,Z25,maxR,step,i,1) 
                
                pyplot.plot(distances, np.exp(mean), color=col,
                            linewidth=2, linestyle='-', label=gmpe)
                
                # Plot Sigma                
                if not Nstd==0:
                    pyplot.plot(distances, np.exp(mean+Nstd*std[0]),
                                linewidth=0.75, color=col, linestyle='--')
                    pyplot.plot(distances, np.exp(mean-Nstd*std[0]),
                                linewidth=0.75, color=col, linestyle='-.')

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

    pyplot.legend(loc="center left", bbox_to_anchor=(1.1, 1.05), fontsize='16')
    pyplot.savefig(os.path.join(output_directory,name) + '_TrellisPlots_Vs30_'
                   + str(Vs30) +'.png', bbox_inches='tight',dpi=200,
                   pad_inches = 0.2)
    pyplot.show()
    pyplot.tight_layout()
    
def compute_matrix_gmpes(imt_list, mag_list, gmpe_list, rake, strike,
                         dip, depth, Z1, Z25, Vs30, region,  maxR,  aratio):
    """
    Compute matrix of median ground-motion predictions for each gmpe for the
    given run configuration for use within Euclidean distance matrix plots,
    Sammons Mapping and hierarchical clustering plots
    """
    step = 1
    Npoints=maxR/step
    if  Z1 == -999:
        Z1 = _get_z1(Vs30,region)

    if  Z25 == -999:
        Z25 = _get_z25(Vs30,region)
    
    # define the matrix with the spectral accelerations
    mtxs_median = {}
 
    for n, i in enumerate(imt_list): #iterate though imt_list

        matrix_medians=np.zeros((len(gmpe_list),(len(mag_list)*int((
            maxR/step)))))
        matrix_sigmas=np.zeros((len(gmpe_list),(len(mag_list)*int((
            maxR/step)))))

        for g, gmpe in enumerate(gmpe_list): 
            medians = []
            sigmas = []
            for l, m in enumerate(mag_list): #iterate though mag_list
        
                strike_g, dip_g, depth_g, aratio_g = _param_gmpes(
                    gmpe,strike, dip, depth[l], aratio, rake) 

                gmm = valid.gsim(gmpe)
                
                mean, std, distances = att_curves(gmm,depth[l],m,aratio_g,
                                                  strike_g,dip_g,rake,Vs30,Z1,
                                                  Z25,maxR,step,i,1) 

                medians = np.append(medians,(np.exp(mean)))
                sigmas = np.append(sigmas,std[0])

            matrix_medians[:][g]= medians
        mtxs_median[n] = matrix_medians
    
    return mtxs_median

def plot_euclidean_util(imt_list, gmpe_list, mtxs, namefig):
    """
    Plot Euclidean distance matrices for given run configuration
    :param imt_list:
        A list e.g. ['PGA', 'SA(0.1)', 'SA(1.0)']
    :param gmpe_list:
        A list e.g. ['BooreEtAl2014', 'CauzziEtAl2014']
    :param mtxs:
        Matrix of median and sigma for each gmpe per imt 
   :param namefig:
        filename for outputted figure 
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
        ax.set_title(i, fontsize='12')

        ticks_loc = ax.get_yticks().tolist()
        ax.xaxis.set_ticks([n for n in range(len(gmpe_list))])
        ax.xaxis.set_ticklabels(gmpe_list,rotation=40)
        ax.yaxis.set_ticks([n for n in range(len(gmpe_list))])
        ax.yaxis.set_ticklabels(gmpe_list)

    pyplot.savefig(namefig, bbox_inches='tight',dpi=200,pad_inches = 0.2)
    pyplot.show()
    pyplot.tight_layout()
    
    return matrix_Dist
    
def plot_sammons_util(imt_list, gmpe_list, mtxs, namefig):
    """
    Plot Sammons maps for given run configuration
    :param imt_list:
        A list e.g. ['PGA', 'SA(0.1)', 'SA(1.0)']
    :param gmpe_list:
        A list e.g. ['BooreEtAl2014', 'CauzziEtAl2014']
    :param mtxs:
        Matrix of median and sigma for each gmpe per imt 
   :param namefig:
        filename for outputted figure 
    """
    # Plots: color for GMPEs
    colors=['r', 'g', 'b', 'y','lime','dodgerblue', 'k', 'gold']
    
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
            label=type(gmpe).__name__
            pyplot.plot(coo[g,0], coo[g,1], 'o', markersize=9, color=colors[
                g], label=gmpe)
            texts.append(pyplot.text(coo[g, 0]+np.abs(coo[g, 0])*0.02,
                                     coo[g, 1]+np.abs(coo[g, 1])*0.,
                                     gmpe_list[g], ha='left', color=col))

        pyplot.title(str(i), fontsize='16')
        pyplot.grid(axis='both', which='both', alpha=0.5)

    pyplot.legend(loc="center left", bbox_to_anchor=(1.1, 0.80), fontsize='16')
    pyplot.savefig(namefig, bbox_inches='tight',dpi=200,pad_inches = 0.2)
    pyplot.show()
    pyplot.tight_layout()
    
    return coo

def plot_cluster_util(imt_list, gmpe_list, mtxs, namefig):
    """
    Plot hierarchical clusters for given run configuration
    :param imt_list:
        A list e.g. ['PGA', 'SA(0.1)', 'SA(1.0)']
    :param gmpe_list:
        A list e.g. ['BooreEtAl2014', 'CauzziEtAl2014']
    :param mtxs:
        Matrix of median and sigma for each gmpe per imt 
   :param namefig:
        filename for outputted figure 
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
        ax.set_xlabel('Euclidean Distance', fontsize = '14')
        ax.set_title(i, fontsize = '14')

    pyplot.savefig(namefig, bbox_inches='tight',dpi=200,pad_inches = 0.2)
    pyplot.show()
    pyplot.tight_layout() 
    
    return matrix_Z