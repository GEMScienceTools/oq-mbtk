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
from openquake.smt.comparison.utils_compare_gmpes import plot_trellis_util,plot_cluster_util, plot_sammons_util, plot_euclidean_util, compute_matrix_gmpes

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
        self.Nstd = config_file['general']['Nstd']
        
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
            'depths']
        for depth_val in range(0,len(self.trellis_depth)):
            self.trellis_depth[depth_val] = float(self.trellis_depth[depth_val])
            
        # Depths per magnitude for use in other functions
        depth_array = {}
        interval_per_mag = (mag_params['mmax'] - mag_params[
            'mmin'])/mag_params['spacing']/(mag_params['mmax'] - mag_params[
                'mmin'])
        for mag, mag_string in enumerate(self.trellis_mag_list):
            depth_array[mag_string] = np.full(int(interval_per_mag),
                                              config_file['source_properties'][
                                                  'depths'][mag])
        self.depth_for_non_trellis_functions = pd.DataFrame(depth_array).melt(
            )['value']
        
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
    plot_trellis_util(self.rake, self.strike, self.dip, self.trellis_depth,
                      self.Z1, self.Z25, self.Vs30, self.region, self.imt_list,
                      self.trellis_mag_list, self.maxR, self.gmpes_list,
                      self.aratio, self.Nstd, self.name_out, output_directory) 
                
def plot_cluster(self, output_directory):
    """
    Plot hierarchical clusters for given configurations
    """
    if len(self.gmpes_list) < 2:
        raise ValueError("Cannot perform clustering for a single GMPE.")   
        
    mtxs_medians = compute_matrix_gmpes(self.imt_list, self.mag_list,
                                            self.gmpes_list, self.rake,
                                            self.strike, self.dip, 
                                            self.depth_for_non_trellis_functions,
                                            self.Z1, self.Z25, self.Vs30,
                                            self.region, self.maxR, self.aratio)
        
    plot_cluster_util(self.imt_list, self.gmpe_labels, mtxs_medians,
                      os.path.join(output_directory,self.name_out) +
                      '_Clustering_Vs30_' + str(self.Vs30) +'.png')
                    
def plot_sammons(self, output_directory):
    """
    Plot Sammons Maps for given configurations
    """        
    if len(self.gmpes_list) < 2:
        raise ValueError("Cannot perform Sammons Mapping for a single GMPE.")
        
    mtxs_medians = compute_matrix_gmpes(self.imt_list, self.mag_list,
                                        self.gmpes_list, self.rake,
                                        self.strike, self.dip, 
                                        self.depth_for_non_trellis_functions,
                                        self.Z1, self.Z25, self.Vs30, 
                                        self.region, self.maxR, self.aratio)
        
    plot_sammons_util(self.imt_list, self.gmpe_labels, mtxs_medians,
                      os.path.join(output_directory,self.name_out) +
                      '_SammonMaps_Vs30_' + str(self.Vs30)+'.png')
        
def plot_euclidean(self,output_directory):
    """
    Plot Euclidean distance matrix for given configurations
    """
    if len(self.gmpes_list) < 2:
        raise ValueError("Cannot perform Euclidean distance matrix plotting for a single GMPE.")
        
    mtxs_medians = compute_matrix_gmpes(self.imt_list, self.mag_list,
                                        self.gmpes_list, self.rake,
                                        self.strike, self.dip,
                                        self.depth_for_non_trellis_functions,
                                        self.Z1, self.Z25, self.Vs30,
                                        self.region, self.maxR, self.aratio)
        
    plot_euclidean_util(self.imt_list, self.gmpe_labels, mtxs_medians,
                        os.path.join(output_directory,self.name_out) 
                        + '_Euclidean_Vs30_' + str(self.Vs30) +'.png')