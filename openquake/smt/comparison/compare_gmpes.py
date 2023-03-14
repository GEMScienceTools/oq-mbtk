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
Module to compare GMPEs using trellis plots, hierarchical clustering, Sammons
Maps and Euclidean distance matrix plots
"""
import numpy as np
import re
import pandas as pd
import copy
import toml
import os

from openquake.hazardlib.imt import from_string
from openquake.hazardlib import valid
from openquake.smt.comparison.utils_compare_gmpes import plot_trellis_util, \
    plot_spectra_util, plot_cluster_util, plot_sammons_util, plot_euclidean_util,\
        compute_matrix_gmpes

class Configurations(object):
    """
    Class to derive configuration for input into GMPE comparison plots
    """
    def __init__(self, filename):
        """
        :param  filename:
            toml file providing configuration for use within comparative
            plotting methods.
        """
        # Import parameters for comparative plots from .toml file
        config_file = toml.load(filename) 
        
        # Get input params from .toml file
        self.name_out = config_file['general']['config_name']
        self.region = config_file['general']['region']
        self.maxR = config_file['general']['maxR']
        self.dist_list = config_file['general']['dist_list']
        self.Nstd = config_file['general']['Nstd']
        self.max_period = config_file['general']['max_period']
        
        self.Vs30 = config_file['site_properties']['vs30']
        self.Z1 = config_file['site_properties']['Z1']
        self.Z25 = config_file['site_properties']['Z25']
        
        self.strike = config_file['source_properties']['strike']
        self.dip = config_file['source_properties']['dip']
        self.rake = config_file['source_properties']['rake']
        
        self.aratio = -999
        
        # One set of magnitudes for use in trellis plots
        self.trellis_mag_list = config_file['source_properties'][
            'trellis_mag_list']
        for mag in range(0,len(self.trellis_mag_list)):
                self.trellis_mag_list[mag] = float(
                    self.trellis_mag_list[mag])
        
        # One set of magnitudes for use in other functions 
        mag_params = config_file['mag_values_non_trellis_functions']
        mag_array = np.arange(mag_params['mmin'],mag_params['mmax'],
                              mag_params['spacing'])
        self.mag_list =  mag_array
        
        # Depths per magnitude for trellis plots
        self.trellis_depth = config_file['source_properties'][
            'trellis_depths']
        for depth_val in range(0,len(self.trellis_depth)):
            self.trellis_depth[depth_val] = float(self.trellis_depth[depth_val])
            
        # Create depth array for non trellis functions 
        non_trellis_depths = pd.DataFrame(config_file[
            'mag_values_non_trellis_functions']['non_trellis_depths'],
            columns=['mag','depth'])
        
        # Round each mag interval to closest integer for depth assignment
        mag_to_nearest_int = pd.Series(dtype='float')
        for mag in mag_array:
            mag_to_nearest_int[mag] = np.round(mag+0.001)
        
        # Assign depth to closest integer
        depth_array_initial = []
        for mag in mag_to_nearest_int:
            for idx in range(0,len(non_trellis_depths['mag'])):
                if mag == non_trellis_depths['mag'][idx]:
                    depth_to_store = non_trellis_depths['depth'][idx]
            depth_array_initial.append(depth_to_store)
            
        self.depth_for_non_trellis_functions = pd.Series(depth_array_initial) 
        
        # Get imts
        self.imt_list = config_file['general']['imt_list']
        for imt in range(0,len(self.imt_list)):
            self.imt_list[imt] = from_string(str(self.imt_list[imt]))  
        
        # Get model labels
        self.gmpe_labels = config_file['gmpe_labels']['gmpes_label']
        
        # Get models
        gmpes_list_initial = []
        config = copy.deepcopy(config_file)
        for key in config['models']:
        # If the key contains a number we take the second part
            if re.search("^\\d+\\-", key):
                tmp = re.sub("^\\d+\\-", "", key)
                value = f"[{tmp}] "
            else:
                value = f"[{key}] "
            if len(config['models'][key]):
               config['models'][key].pop('style', None)
               value += '\n' + str(toml.dumps(config['models'][key]))
            gmpes_list_initial.append(valid.gsim(value))
            
        self.gmpes_list = []
        for gmpe in range(0,len(gmpes_list_initial)):
            self.gmpes_list.append(str(gmpes_list_initial[gmpe]))
        
def plot_trellis(self, output_directory):
    """
    Plot trellis for given run configuration
    """
    if len(self.gmpes_list) > 8:
        raise ValueError("Cannot plot more than 8 GMPEs at once.") 
        
    plot_trellis_util(self.rake, self.strike, self.dip, self.trellis_depth,
                      self.Z1, self.Z25, self.Vs30, self.region, self.imt_list,
                      self.trellis_mag_list, self.maxR, self.gmpes_list,
                      self.aratio, self.Nstd, self.name_out, output_directory) 
                
def plot_spectra(self, output_directory):
    """
    Plot response spectra and GMPE sigma wrt spectral period for given run
    configuration
    """
    if len(self.gmpes_list) > 8:
        raise ValueError("Cannot plot more than 8 GMPEs at once.") 
        
    plot_spectra_util(self.rake, self.strike, self.dip, self.trellis_depth,
                      self.Z1, self.Z25, self.Vs30, self.region,
                      self.max_period, self.trellis_mag_list,
                      self.dist_list, self.gmpes_list, self.aratio, self.Nstd,
                      self.name_out, output_directory) 

def plot_cluster(self, output_directory):
    """
    Plot hierarchical clusters of (1) median and (2) median +1 sigma (i.e. 84th
    percentile) predicted ground-motion by each GMPE for given configurations
    """
    if len(self.gmpes_list) > 8:
        raise ValueError("Cannot plot more than 8 GMPEs at once.") 
        
    if len(self.gmpes_list) < 2:
        raise ValueError("Cannot perform clustering for a single GMPE.")   

    # Cluster median predicted ground-motion
    mtxs_medians = compute_matrix_gmpes(self.imt_list, self.mag_list,
                                            self.gmpes_list, self.rake,
                                            self.strike, self.dip, 
                                            self.depth_for_non_trellis_functions,
                                            self.Z1, self.Z25, self.Vs30,
                                            self.region, self.maxR, self.aratio,
                                            mtxs_type = 'median')

    mtxs_plus_1_sigma = compute_matrix_gmpes(self.imt_list, self.mag_list,
                                            self.gmpes_list, self.rake,
                                            self.strike, self.dip, 
                                            self.depth_for_non_trellis_functions,
                                            self.Z1, self.Z25, self.Vs30,
                                            self.region, self.maxR, self.aratio,
                                            mtxs_type = '+1_sigma')
    
    # Cluster median + 1 sigma
    plot_cluster_util(self.imt_list, self.gmpe_labels, mtxs_medians,
                      os.path.join(output_directory,self.name_out) +
                      '_Median_Clustering_Vs30_' + str(self.Vs30) +'.png',
                      mtxs_type = 'median')    
    
    plot_cluster_util(self.imt_list, self.gmpe_labels, mtxs_plus_1_sigma,
                      os.path.join(output_directory,self.name_out) +
                      '_+1_sigma_Clustering_Vs30_' + str(self.Vs30) +'.png',
                      mtxs_type = '+1_sigma')                    

def plot_sammons(self, output_directory):
    """
    Plot Sammons Maps of median predicted ground-motion by each GMPE for given
    configurations
    """ 
    if len(self.gmpes_list) > 8:
        raise ValueError("Cannot plot more than 8 GMPEs at once.") 
        
    if len(self.gmpes_list) < 2:
        raise ValueError("Cannot perform Sammons Mapping for a single GMPE.")
        
    mtxs_medians = compute_matrix_gmpes(self.imt_list, self.mag_list,
                                        self.gmpes_list, self.rake,
                                        self.strike, self.dip, 
                                        self.depth_for_non_trellis_functions,
                                        self.Z1, self.Z25, self.Vs30, 
                                        self.region, self.maxR, self.aratio,
                                        mtxs_type = 'median')
        
    plot_sammons_util(self.imt_list, self.gmpe_labels, mtxs_medians,
                      os.path.join(output_directory,self.name_out) +
                      '_SammonMaps_Vs30_' + str(self.Vs30)+'.png')
        
def plot_euclidean(self,output_directory):
    """
    Plot Euclidean distance matrix of median predicted ground-motion by each GMPE
    for given configurations
    """
    if len(self.gmpes_list) > 8:
        raise ValueError("Cannot plot more than 8 GMPEs at once.") 
        
    if len(self.gmpes_list) < 2:
        raise ValueError("Cannot perform Euclidean distance matrix plotting for a single GMPE.")
        
    mtxs_medians = compute_matrix_gmpes(self.imt_list, self.mag_list,
                                        self.gmpes_list, self.rake,
                                        self.strike, self.dip,
                                        self.depth_for_non_trellis_functions,
                                        self.Z1, self.Z25, self.Vs30,
                                        self.region, self.maxR, self.aratio,
                                        mtxs_type = 'median')
        
    plot_euclidean_util(self.imt_list, self.gmpe_labels, mtxs_medians,
                        os.path.join(output_directory,self.name_out) 
                        + '_Euclidean_Vs30_' + str(self.Vs30) +'.png')