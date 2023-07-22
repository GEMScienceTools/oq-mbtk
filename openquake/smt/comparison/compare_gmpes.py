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
        self.region = config_file['general']['region']
        self.eshm20_region = config_file['general']['eshm20_region']
        self.maxR = config_file['general']['maxR']
        self.dist_list = config_file['general']['dist_list']
        self.Nstd = config_file['general']['Nstd']
        self.max_period = config_file['general']['max_period']
        
        self.custom_color_flag = config_file['custom_colors']['custom_colors_flag']
        self.custom_color_list = config_file['custom_colors']['custom_colors_list']
        
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

        # Check number of GMPEs matches number of GMPE labels
        if len(self.gmpes_list) != len(self.gmpe_labels):
            raise ValueError("Number of labels must match number of GMPEs.")

        # If weight is assigned to a GMPE get it + check sum of weights for 
        # GMPEs with weights allocated = 1 (up to 2 GMC logic trees max)
        get_weights_gmc1 = {}
        get_weights_gmc2 = {}
        for gmpe in self.gmpes_list:
            if 'lt_weight' in gmpe:
                split_gmpe_str = str(gmpe).splitlines()
                for idx, component in enumerate(split_gmpe_str):
                    if 'lt_weight_gmc1' in component:
                        get_weights_gmc1[gmpe] = float(split_gmpe_str[
                            idx].split('=')[1])
                    if 'lt_weight_gmc2' in component:
                        get_weights_gmc2[gmpe] = float(split_gmpe_str[
                            idx].split('=')[1])
            else:
                pass
            
        # Check weights for each logic tree (if present) equal 1.0
        if get_weights_gmc1 != {}:
            check_weights_gmc1 = np.array(pd.Series(get_weights_gmc1))
            if np.sum(check_weights_gmc1, axis = 0) != 1.0:
                raise ValueError("GMPE logic tree weights must total 1.0")
            self.lt_weights_gmc1 = get_weights_gmc1
        else:
            self.lt_weights_gmc1 = None
        
        if get_weights_gmc2 != {}:
            check_weights_gmc2 = np.array(pd.Series(get_weights_gmc2))
            if np.sum(check_weights_gmc2, axis = 0) != 1.0:
                raise ValueError("GMPE logic tree weights must total 1.0")
            self.lt_weights_gmc2 = get_weights_gmc2
        else:
            self.lt_weights_gmc2 = None

def plot_trellis(filename, output_directory):
    """
    Plot trellis for given run configuration
    :param  filename:
        toml file providing configuration for use within comparative
        plotting methods.
    """ 
    # Generate config object
    config = Configurations(filename)
    
    plot_trellis_util(config.rake, config.strike, config.dip,
                      config.trellis_depth, config.Z1, config.Z25, config.Vs30,
                      config.region, config.imt_list, config.trellis_mag_list,
                      config.maxR, config.gmpes_list, config.aratio,
                      config.Nstd, output_directory, config.custom_color_flag,
                      config.custom_color_list, config.eshm20_region,
                      config.lt_weights_gmc1, config.lt_weights_gmc2) 
                
def plot_spectra(filename, output_directory, obs_spectra = None):
    """
    Plot response spectra and GMPE sigma wrt spectral period for given run
    configuration
    :param  filename:
        toml file providing configuration for use within comparative
        plotting methods.
    :param obs_spectra:
        csv of an observed spectra to plot and associated event information.
        An example file can be found in openquake.smt.tests.file_samples.
    """
    # Generate config object
    config = Configurations(filename)
    
    # Get observed spectra information if obs_spectra
    if obs_spectra is not None:
        obs_spectra = pd.read_csv(obs_spectra)
    else:
        obs_spectra = None
    
    plot_spectra_util(config.rake, config.strike, config.dip,
                      config.trellis_depth, config.Z1, config.Z25, config.Vs30,
                      config.region, config.max_period, config.trellis_mag_list,
                      config.dist_list, config.gmpes_list, config.aratio,
                      config.Nstd, output_directory, config.custom_color_flag,
                      config.custom_color_list, config.eshm20_region,
                      config.lt_weights_gmc1, config.lt_weights_gmc2, obs_spectra) 

def plot_cluster(filename, output_directory):
    """
    Plot hierarchical clusters of (1) median, (2) 84th percentile and (3) 16th
    percentile of predicted ground-motion by each GMPE for given configurations
    :param  filename:
        toml file providing configuration for use within comparative
        plotting methods.
    """ 
    # Generate config object with each set of run parameters
    config = Configurations(filename)
    
    if len(config.gmpes_list) < 2:
        raise ValueError("Cannot perform clustering for a single GMPE.")   

    # Cluster median predicted ground-motion
    mtxs_medians = compute_matrix_gmpes(config.imt_list, config.mag_list,
                                            config.gmpes_list, config.rake,
                                            config.strike, config.dip, 
                                            config.depth_for_non_trellis_functions,
                                            config.Z1, config.Z25, config.Vs30,
                                            config.region, config.maxR, config.aratio,
                                            config.eshm20_region,mtxs_type='median')

    mtxs_84th_perc = compute_matrix_gmpes(config.imt_list, config.mag_list,
                                            config.gmpes_list, config.rake,
                                            config.strike, config.dip, 
                                            config.depth_for_non_trellis_functions,
                                            config.Z1, config.Z25, config.Vs30,
                                            config.region, config.maxR, config.aratio,
                                            config.eshm20_region,mtxs_type='84th_perc')
    
    mtxs_16th_perc = compute_matrix_gmpes(config.imt_list, config.mag_list,
                                            config.gmpes_list, config.rake,
                                            config.strike, config.dip, 
                                            config.depth_for_non_trellis_functions,
                                            config.Z1, config.Z25, config.Vs30,
                                            config.region, config.maxR, config.aratio,
                                            config.eshm20_region,mtxs_type='16th_perc')
    
    # Cluster by median
    plot_cluster_util(config.imt_list, config.gmpe_labels, mtxs_medians,
                      os.path.join(output_directory,'Median_Clustering.png'),
                      mtxs_type = 'median')    
    
    # Cluster by 84th percentile
    plot_cluster_util(config.imt_list, config.gmpe_labels, mtxs_84th_perc,
                      os.path.join(output_directory,'84th_perc_Clustering.png'),
                      mtxs_type = '84th_perc')                    

    # Cluster by 16th percentile
    plot_cluster_util(config.imt_list, config.gmpe_labels, mtxs_16th_perc,
                      os.path.join(output_directory,'16th_perc_Clustering.png'),
                      mtxs_type = '16th_perc')  

def plot_sammons(filename, output_directory):
    """
    Plot Sammons Maps of median and 84th percentile predicted ground-motion
    by each GMPE for given configurations
    :param  filename:
        toml file providing configuration for use within comparative
        plotting methods.
    """ 
    # Generate config object with each set of run parameters
    config = Configurations(filename)
    
    if len(config.gmpes_list) < 2:
        raise ValueError("Cannot perform Sammons Mapping for a single GMPE.")
        
    mtxs_medians = compute_matrix_gmpes(config.imt_list, config.mag_list,
                                        config.gmpes_list, config.rake,
                                        config.strike, config.dip, 
                                        config.depth_for_non_trellis_functions,
                                        config.Z1, config.Z25, config.Vs30, 
                                        config.region, config.maxR, config.aratio,
                                        config.eshm20_region,mtxs_type='median')
    
    mtxs_84th_perc = compute_matrix_gmpes(config.imt_list, config.mag_list,
                                            config.gmpes_list, config.rake,
                                            config.strike, config.dip, 
                                            config.depth_for_non_trellis_functions,
                                            config.Z1, config.Z25, config.Vs30,
                                            config.region, config.maxR, config.aratio,
                                            config.eshm20_region,mtxs_type='84th_perc')
    
    mtxs_16th_perc = compute_matrix_gmpes(config.imt_list, config.mag_list,
                                            config.gmpes_list, config.rake,
                                            config.strike, config.dip, 
                                            config.depth_for_non_trellis_functions,
                                            config.Z1, config.Z25, config.Vs30,
                                            config.region, config.maxR, config.aratio,
                                            config.eshm20_region,mtxs_type='16th_perc')
    
    plot_sammons_util(config.imt_list, config.gmpe_labels, mtxs_medians,
                      os.path.join(output_directory,'Median_SammonMaps.png'),
                      config.custom_color_flag, config.custom_color_list,
                      mtxs_type = 'median')
    
    plot_sammons_util(config.imt_list, config.gmpe_labels, mtxs_84th_perc,
                      os.path.join(output_directory,'84th_perc_SammonMaps.png'),
                      config.custom_color_flag, config.custom_color_list,
                      mtxs_type = '84th_perc')
    
    plot_sammons_util(config.imt_list, config.gmpe_labels, mtxs_16th_perc,
                      os.path.join(output_directory,'16th_perc_SammonMaps.png'),
                      config.custom_color_flag, config.custom_color_list,
                      mtxs_type = '16th_perc')
   
def plot_euclidean(filename, output_directory):
    """
    Plot Euclidean distance matrix of median and 84th percentile predicted
    ground-motion by each GMPE for given configurations
    :param  filename:
        toml file providing configuration for use within comparative
        plotting methods.    
    """ 
    # Generate config object
    config = Configurations(filename)
    
    if len(config.gmpes_list) < 2:
        raise ValueError("Cannot perform Euclidean distance matrix plotting for a single GMPE.")
        
    mtxs_medians = compute_matrix_gmpes(config.imt_list, config.mag_list,
                                        config.gmpes_list, config.rake,
                                        config.strike, config.dip,
                                        config.depth_for_non_trellis_functions,
                                        config.Z1, config.Z25, config.Vs30,
                                        config.region, config.maxR, config.aratio,
                                        config.eshm20_region,mtxs_type='median')
    
    mtxs_84th_perc = compute_matrix_gmpes(config.imt_list, config.mag_list,
                                            config.gmpes_list, config.rake,
                                            config.strike, config.dip, 
                                            config.depth_for_non_trellis_functions,
                                            config.Z1, config.Z25, config.Vs30,
                                            config.region, config.maxR, config.aratio,
                                            config.eshm20_region,mtxs_type='84th_perc')
    
    mtxs_16th_perc = compute_matrix_gmpes(config.imt_list, config.mag_list,
                                            config.gmpes_list, config.rake,
                                            config.strike, config.dip, 
                                            config.depth_for_non_trellis_functions,
                                            config.Z1, config.Z25, config.Vs30,
                                            config.region, config.maxR, config.aratio,
                                            config.eshm20_region,mtxs_type='16th_perc')
    plot_euclidean_util(config.imt_list, config.gmpe_labels, mtxs_medians,
                        os.path.join(output_directory,'Median_Euclidean.png'),
                        mtxs_type = 'median')
    
    plot_euclidean_util(config.imt_list, config.gmpe_labels, mtxs_84th_perc,
                        os.path.join(output_directory,'84th_perc_Euclidean.png'),
                        mtxs_type = '84th_perc')
    
    plot_euclidean_util(config.imt_list, config.gmpe_labels, mtxs_16th_perc,
                        os.path.join(output_directory,'16th_perc_Euclidean.png'),
                        mtxs_type = '16th_perc')