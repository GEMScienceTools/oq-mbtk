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
    plot_trellis_util, plot_spectra_util, plot_ratios_util,
    plot_cluster_util, plot_sammons_util, plot_euclidean_util,
    compute_matrix_gmpes)


# If a param is not in toml, use this value instead for the sites
SITE_OPTIONAL = {
    "z1pt0": -999, # Let param be computed using each GMM's vs30 to z1pt0
    "z2pt5": -999, # Let param be computed using each GMM's vs30 to z2pt5
    "up_or_down_dip": 1, # Assume site is up-dip
    "volc_back_arc": False, # Asssume site is not in back-arc
    "eshm20_region": 0}  # Assume default region for ESHM version of Kotha et al. (2020)


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
        self.minR = config_file['general']['minR']
        self.maxR = config_file['general']['maxR']
        self.dist_type = config_file['general']['dist_type']
        self.dist_list = config_file['general']['dist_list']
        self.Nstd = config_file['general']['Nstd']
        self.max_period = config_file['general']['max_period']
        
        # Get site params
        self.vs30 = config_file['site_properties']['vs30'] # Must be provided
        for par in SITE_OPTIONAL:
            if par not in config_file['site_properties']:
                setattr(self, par, SITE_OPTIONAL[par]) # Assign default if not provided
            else:
                setattr(self, par, config_file['site_properties'][par])

        # Get source params
        self.strike = config_file['source_properties']['strike']
        self.dip = config_file['source_properties']['dip']
        self.rake = config_file['source_properties']['rake']
        self.mag_list = np.array(config_file['source_properties']['mags'])
        self.depth_list = np.array(config_file['source_properties']['depths'])
        self.ztor = config_file['source_properties']['ztor']
        self.aratio = config_file['source_properties']['aratio']
        self.trt = config_file['source_properties']['trt']
        
        # Get custom colors
        self.custom_color_flag = config_file['custom_colors']['custom_colors_flag']
        self.custom_color_list = config_file['custom_colors']['custom_colors_list']
        
        # Check same length mag and depth lists to avoid indexing error
        assert len(self.mag_list) == len(self.depth_list)
        
        # Get mags and depths for Sammons, Euclidean distance and clustering
        self.mags_euclidean, self.depths_euclidean = self.get_eucl_mags_deps(config_file)
        self.gmpe_labels = config_file['euclidean_analysis']['gmpe_labels']

        # Get imts
        self.imt_list = [from_string(imt) for imt in config_file['general']['imt_list']]

        # Get GMMs
        self.gmpes_list, self.baseline_gmm = self.get_gmpes(config_file)

        # Get lt weights
        (self.lt_weights_gmc1, self.lt_weights_gmc2, self.lt_weights_gmc3,
         self.lt_weights_gmc4) = self.get_lt_weights(self.gmpes_list)

    def get_gmpes(self, config_file):
        """
        Extract strings of the GMPEs from the configuration file. Also get the 
        labels used in Sammons maps, Clustering (dendrograms) and Euclidean distance
        matrix plots. The baseline GMM for computing ratios with is also extracted
        if specified within the toml file.
        """
        # Get the GMPEs
        gmpe_list = []
        config = copy.deepcopy(config_file)
        for key in config['models']:
            value = self.get_gmm(key, config['models'])
            gmpe_list.append(value)

        # Get the baseline GMPE used to compute ratios of GMPEs with if required
        if 'ratios_baseline_gmm' in config_file.keys():
            if len(config_file['ratios_baseline_gmm']) > 1:
                raise ValueError('Only one baseline GMPE should be specified.')
            for key in config_file['ratios_baseline_gmm']:
                baseline_gmm = self.get_gmm(key, config['ratios_baseline_gmm'])
        else:
            baseline_gmm = None

        return gmpe_list, baseline_gmm

    def get_gmm(self, key, models):
        """
        Get the model from the toml in the string format required to create an
        OpenQuake gsim object from within mgmpe_check (in utils_gmpes.py)
        """
        # If the key contains a number we take the second part
        if re.search("^\\d+\\-", key):
            tmp = re.sub("^\\d+\\-", "", key)
            value = f"[{tmp}] "
        else:
            value = f"[{key}] "
        if len(models[key]):
            models[key].pop('style', None)
            value += '\n' + str(toml.dumps(models[key]))
            
        return value.strip()

    def get_lt_weights(self, gmpe_list):
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

    def get_eucl_mags_deps(self, config_file):
        """
        For each magnitude considered within the Sammons Maps, Euclidean distance
        matrix plots and agglomerative clustering dendrograms get the magnitudes
        and assign a depth for each.
        """
        # Make array of the magnitudes
        mag_params = config_file['euclidean_analysis']
        mags_euclidean = np.arange(
            mag_params['mmin'], mag_params['mmax'], mag_params['spacing'])

        # Create dataframe of depth to assign per mag bin
        depths_for_euclidean = pd.DataFrame(config_file[
            'euclidean_analysis']['depths_for_euclidean'], columns=['mag','depth'])
                
        # Round each mag in mag_array to closest integer
        mag_to_nearest_int = pd.Series(dtype='float')
        for mag in mags_euclidean:
            mag_to_nearest_int[mag] = np.round(mag+0.001)

        # Assign depth to each mag in mag_array using rounded mags
        depths_euclidean = []
        for idx_mag, rounded_mag in enumerate(mag_to_nearest_int):
            for idx, mag in enumerate(depths_for_euclidean['mag']):
                if rounded_mag == mag:
                    depth_to_store = depths_for_euclidean['depth'][idx]
                    depths_euclidean.append(depth_to_store)
        
        return  mags_euclidean, pd.Series(depths_euclidean)


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

                
def plot_spectra(filename, output_directory, obs_spectra_fname=None):
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

    if obs_spectra_fname is not None:
        try:
            assert len(config.mag_list) == 1 and len(config.depth_list) == 1
        except:
            raise ValueError("If plotting an observed spectra you must only " \
                             "specify 1 magnitude and depth combination for " \
                             "response spectra plotting in the toml file.")

    store_gmc_lts = plot_spectra_util(config, output_directory, obs_spectra_fname)

    return store_gmc_lts


def plot_ratios(filename, output_directory):
    """
    Plot ratio (GMPE median attenuation/baseline GMPE median attenuation) for
    given run configuration
    :param  filename:
        toml file providing configuration for use within comparative
        plotting methods.
    """ 
    config = Configurations(filename)

    if config.baseline_gmm is None:
        raise ValueError(
            'User must specify a baseline GMPE to generate ratio plots')
    
    plot_ratios_util(config, output_directory)


def plot_cluster(filename, output_directory):
    """
    Plot hierarchical clusters of (1) median, (2) 84th percentile and (3) 16th
    percentile of predicted ground-motion by each GMPE for given configurations
    :param  filename:
        toml file providing configuration for use within comparative
        plotting methods.
    """ 
    config = Configurations(filename)
    
    if len(config.gmpes_list) != len(config.gmpe_labels):
        raise ValueError("Number of labels must match number of GMPEs.")

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

    if len(config.gmpes_list) != len(config.gmpe_labels):
        raise ValueError("Number of labels must match number of GMPEs.")
    
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
    
    if len(config.gmpes_list) != len(config.gmpe_labels):
        raise ValueError("Number of labels must match number of GMPEs.")
    
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