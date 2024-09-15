# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2024 GEM Foundation
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
from openquake.smt.comparison.utils_compare_gmpes import (
    plot_trellis_util, plot_spectra_util, plot_cluster_util, plot_sammons_util,
    plot_euclidean_util, compute_matrix_gmpes, plot_ratios_util)


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
        
        # Get general params
        self.eshm20_region = config_file['general']['eshm20_region']
        self.minR = config_file['general']['minR']
        self.maxR = config_file['general']['maxR']
        self.dist_type = config_file['general']['dist_type']
        self.dist_list = config_file['general']['dist_list']
        self.Nstd = config_file['general']['Nstd']
        self.max_period = config_file['general']['max_period']
        
        # Get site params
        self.Vs30 = config_file['site_properties']['vs30']
        self.Z1 = config_file['site_properties']['Z1']
        self.Z25 = config_file['site_properties']['Z25']
        up_or_down_dip = config_file['site_properties']['up_or_down_dip']
        self.up_or_down_dip = float(up_or_down_dip)
        self.region = config_file['site_properties']['region']
        
        # Get rupture params
        self.trt = config_file['source_properties']['trt']
        if self.trt == 'None':
            self.trt = None
        self.ztor = config_file['source_properties']['ztor']
        if self.ztor == 'None':
            self.ztor = None
        self.strike = config_file['source_properties']['strike']
        self.dip = config_file['source_properties']['dip']
        self.rake = config_file['source_properties']['rake']
        self.aratio = config_file['source_properties']['aratio']
        
        # Get custom colors
        self.custom_color_flag = config_file['custom_colors'][
            'custom_colors_flag']
        self.custom_color_list = config_file['custom_colors'][
            'custom_colors_list']

        # One set of magnitudes for use in trellis plots
        self.trellis_and_rs_mag_list = config_file['source_properties'][
            'trellis_and_rs_mag_list']
        for idx, mag in enumerate(self.trellis_and_rs_mag_list):
                self.trellis_and_rs_mag_list[idx] = float(
                    self.trellis_and_rs_mag_list[idx])
        
        # Depths per magnitude for trellis plots
        self.trellis_and_rs_depth_list = config_file['source_properties'][
            'trellis_and_rs_depths']
        for idx, depth in enumerate(self.trellis_and_rs_depth_list):
            self.trellis_and_rs_depth_list[idx] = float(
                self.trellis_and_rs_depth_list[idx])
        
        # Get mags for Sammons, Euclidean distance and clustering
        mag_params = config_file['mag_values_non_trellis_or_spectra_functions']
        self.mag_list = np.arange(
            mag_params['mmin'], mag_params['mmax'], mag_params['spacing'])
        
        # Get depths for Sammons, Euclidean distance and clustering
        self.depth_for_non_trel_or_rs_fun = assign_depths_per_mag_bin(
            config_file, self.mag_list)
        
        # Get imts
        self.imt_list = config_file['general']['imt_list']
        for idx, imt in enumerate(self.imt_list):
            self.imt_list[idx] = from_string(str(self.imt_list[idx]))  
        
        # Get models and model labels
        self.gmpe_labels, self.gmpes_list = get_gmpes(config_file)

        # Get lt weights
        (self.lt_weights_gmc1, self.lt_weights_gmc2, self.lt_weights_gmc3,
         self.lt_weights_gmc4) = get_lt_weights(self.gmpes_list)


def get_gmpes(config_file):
    """
    Extract strings of the GMPEs from the configuration file
    """
    gmpe_labels = config_file['gmpe_labels']['gmpes_label']
    gmpe_list = []
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
        value = value.strip()
        gmpe_list.append(value.strip())

    # Check number of GMPEs matches number of GMPE labels
    if len(gmpe_list) != len(gmpe_labels):
        raise ValueError("Number of labels must match number of GMPEs.")

    return gmpe_labels, gmpe_list


def get_lt_weights(gmpe_list):
    """
    Manage the logic tree weight assigned for each GMPE in the toml (if any)
    """
    # If weight is assigned to a GMPE get it + check sum of weights for 
    # GMPEs with weights allocated is about 1
    weights = [{}, {}, {}, {}]
    for gmpe in gmpe_list:
        if 'lt_weight' in gmpe:
            split_gmpe_str = str(gmpe).splitlines()
            for idx, component in enumerate(split_gmpe_str):
                if 'lt_weight_gmc1' in component:
                    weights[0][gmpe] = float(split_gmpe_str[
                        idx].split('=')[1])
                if 'lt_weight_gmc2' in component:
                    weights[1][gmpe] = float(split_gmpe_str[
                        idx].split('=')[1])                       
                if 'lt_weight_gmc3' in component:
                    weights[2][gmpe] = float(split_gmpe_str[
                        idx].split('=')[1])
                if 'lt_weight_gmc4' in component:
                    weights[3][gmpe] = float(split_gmpe_str[
                        idx].split('=')[1])
        
    # Check weights for each logic tree (if present) equal 1.0
    msg = "Sum of GMC logic tree weights must be 1.0"
    if weights[0] != {}:
        check_weights_gmc1 = np.array(pd.Series(weights[0]))
        lt_total_wt_gmc1 = np.sum(check_weights_gmc1, axis=0)
        assert abs(lt_total_wt_gmc1-1.0) < 1e-10, msg
        lt_weights_gmc1 = weights[0]
    else:
        lt_weights_gmc1 = None
    
    if weights[1] != {}:
        check_weights_gmc2 = np.array(pd.Series(weights[1]))
        lt_total_wt_gmc2 = np.sum(check_weights_gmc2, axis=0)
        assert abs(lt_total_wt_gmc2-1.0) < 1e-10, msg
        lt_weights_gmc2 = weights[1]
    else:
        lt_weights_gmc2 = None

    if weights[2] != {}:
        check_weights_gmc3 = np.array(pd.Series(weights[2]))
        lt_total_wt_gmc3 = np.sum(check_weights_gmc3, axis=0)
        assert abs(lt_total_wt_gmc3-1.0) < 1e-10, msg
        lt_weights_gmc3 = weights[2]
    else:
        lt_weights_gmc3 = None
        
    if weights[3] != {}:
        check_weights_gmc4 = np.array(pd.Series(weights[3]))
        lt_total_wt_gmc4 = np.sum(check_weights_gmc4, axis=0)
        assert abs(lt_total_wt_gmc4-1.0) < 1e-10, msg
        lt_weights_gmc4 = weights[3]
    else:
        lt_weights_gmc4 = None

    return lt_weights_gmc1, lt_weights_gmc2, lt_weights_gmc3, lt_weights_gmc4


def assign_depths_per_mag_bin(config_file, mag_array):
    """
    For each magnitude considered within the Sammons Maps, Euclidean distance
    and clustering plots assign a depth
    """
    # Create dataframe of depth to assign per mag bin
    non_trellis_or_spectra_depths = pd.DataFrame(config_file[
        'mag_values_non_trellis_or_spectra_functions'][
            'non_trellis_or_spectra_depths'], columns=['mag','depth'])
            
    # Round each mag in mag_array to closest integer
    mag_to_nearest_int = pd.Series(dtype='float')
    for mag in mag_array:
        mag_to_nearest_int[mag] = np.round(mag+0.001)

    # Assign depth to each mag in mag_array using rounded mags
    depth_array_initial = []
    for idx_mag, rounded_mag in enumerate(mag_to_nearest_int):
        for idx, val in enumerate(non_trellis_or_spectra_depths['mag']):
            if rounded_mag == non_trellis_or_spectra_depths['mag'][idx]:
                depth_to_store = non_trellis_or_spectra_depths['depth'][idx]
                depth_array_initial.append(depth_to_store)
    
    return pd.Series(depth_array_initial) 


def plot_trellis(filename, output_directory):
    """
    Plot trellis for given run configuration
    :param  filename:
        toml file providing configuration for use within comparative
        plotting methods.
    """ 
    config = Configurations(filename)
    
    store_gmm_curves = plot_trellis_util(config, output_directory) 
    
    return store_gmm_curves

                
def plot_spectra(filename, output_directory, obs_spectra=None):
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
    config = Configurations(filename)
    
    store_gmc_lts = plot_spectra_util(config, output_directory, obs_spectra)

    return store_gmc_lts


def plot_cluster(filename, output_directory):
    """
    Plot hierarchical clusters of (1) median, (2) 84th percentile and (3) 16th
    percentile of predicted ground-motion by each GMPE for given configurations
    :param  filename:
        toml file providing configuration for use within comparative
        plotting methods.
    """ 
    config = Configurations(filename)
    
    if len(config.gmpes_list) < 2:
        raise ValueError("Cannot perform clustering for a single GMPE.")   

    # Cluster median predicted ground-motion
    mtxs_50th_perc = compute_matrix_gmpes(config, mtxs_type='median')
    mtxs_84th_perc = compute_matrix_gmpes(config, mtxs_type='84th_perc')
    mtxs_16th_perc = compute_matrix_gmpes(config, mtxs_type='16th_perc')
    
    # Cluster by median
    plot_cluster_util(config.imt_list, config.gmpe_labels, mtxs_50th_perc,
                      os.path.join(output_directory,'Median_Clustering.png'),
                      mtxs_type='median')    
    
    # Cluster by 84th percentile
    plot_cluster_util(config.imt_list, config.gmpe_labels, mtxs_84th_perc,
                      os.path.join(output_directory,'84th_perc_Clustering.png'),
                      mtxs_type='84th_perc')                    

    # Cluster by 16th percentile
    plot_cluster_util(config.imt_list, config.gmpe_labels, mtxs_16th_perc,
                      os.path.join(output_directory,'16th_perc_Clustering.png'),
                      mtxs_type='16th_perc')  


def plot_sammons(filename, output_directory):
    """
    Plot Sammons Maps of median and 84th percentile predicted ground-motion
    by each GMPE for given configurations
    :param  filename:
        toml file providing configuration for use within comparative
        plotting methods.
    """ 
    config = Configurations(filename)
    
    if len(config.gmpes_list) < 2:
        raise ValueError("Cannot perform Sammons Mapping for a single GMPE.")

    mtxs_50th_perc = compute_matrix_gmpes(config, mtxs_type='median')
    
    mtxs_84th_perc = compute_matrix_gmpes(config, mtxs_type='84th_perc')
    
    mtxs_16th_perc = compute_matrix_gmpes(config, mtxs_type='16th_perc')
    
    plot_sammons_util(config.imt_list, config.gmpe_labels, mtxs_50th_perc,
                      os.path.join(output_directory,'Median_SammonMaps.png'),
                      config.custom_color_flag, config.custom_color_list,
                      mtxs_type='median')
    
    plot_sammons_util(config.imt_list, config.gmpe_labels, mtxs_84th_perc,
                      os.path.join(output_directory,'84th_perc_SammonMaps.png'),
                      config.custom_color_flag, config.custom_color_list,
                      mtxs_type='84th_perc')
    
    plot_sammons_util(config.imt_list, config.gmpe_labels, mtxs_16th_perc,
                      os.path.join(output_directory,'16th_perc_SammonMaps.png'),
                      config.custom_color_flag, config.custom_color_list,
                      mtxs_type='16th_perc')
   
    
def plot_euclidean(filename, output_directory):
    """
    Plot Euclidean distance matrix of median and 84th percentile predicted
    ground-motion by each GMPE for given configurations
    :param  filename:
        toml file providing configuration for use within comparative
        plotting methods.    
    """ 
    config = Configurations(filename)
    
    if len(config.gmpes_list) < 2:
        raise ValueError(
            "Cannot perform Euclidean dist matrix plotting for a single GMPE.")
        
    mtxs_50th_perc = compute_matrix_gmpes(config, mtxs_type='median')
    
    mtxs_84th_perc = compute_matrix_gmpes(config, mtxs_type='84th_perc')
    
    mtxs_16th_perc = compute_matrix_gmpes(config, mtxs_type='16th_perc')
    
    plot_euclidean_util(config.imt_list, config.gmpe_labels, mtxs_50th_perc,
                        os.path.join(output_directory,'Median_Euclidean.png'),
                        mtxs_type='median')
    
    plot_euclidean_util(config.imt_list, config.gmpe_labels, mtxs_84th_perc,
                        os.path.join(output_directory,'84th_perc_Euclidean.png'),
                        mtxs_type='84th_perc')
    
    plot_euclidean_util(config.imt_list, config.gmpe_labels, mtxs_16th_perc,
                        os.path.join(output_directory,'16th_perc_Euclidean.png'),
                        mtxs_type='16th_perc')
    

def plot_ratios(filename, output_directory):
    """
    Plot ratios of each GMM/a baseline GMM versus distance for given run
    configuration
    :param  filename:
        toml file providing configuration for use within comparative
        plotting methods.
    """ 
    config = Configurations(filename)
    
    plot_ratios_util(config, output_directory) 