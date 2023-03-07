# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2023 GEM Foundation and G. Weatherill
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
    Class to derive sets of configurations for input into GMPE comparison plots
    """
    def __init__(self, filename):
        """
        :param  filename:
            toml file providing run configurations for use within comparative
            plotting methods.
        """
        # Import parameters for comparative plots from .toml file
        config_file = toml.load(filename) 
        
        # Get input params from .toml file
        self.name_out = pd.Series(config_file['run_names']['name'])
        self.Vs30 = pd.Series(config_file['vs30_values']['vs30'])
        self.Z1 = pd.Series(config_file['z1pt0_values']['Z1'])
        self.Z25 = pd.Series(config_file['z2pt5_values']['Z25'])
        self.region = pd.Series(config_file['region_list']['region'])
        self.strike = pd.Series(config_file['strike_values']['strike'])
        self.dip = pd.Series(config_file['dip_values']['dip'])
        self.rake = pd.Series(config_file['rake_values']['rake'])
        
        # One set of magnitudes for use in trellis plots
        self.trellis_mag_list_per_config = config_file['trellis_mag_values'][
            'trellis_mag_list']
        for config_set in range(0,len(self.trellis_mag_list_per_config)):
            for mag in range(0,len(self.trellis_mag_list_per_config[
                    config_set])):
                self.trellis_mag_list_per_config[config_set][mag] = float(
                    self.trellis_mag_list_per_config[config_set][mag])
        
        # One set of magnitudes for use in other functions 
        mag_params_all_runs = config_file['mag_values']
        mag_array_per_run = {}
        for run in mag_params_all_runs:
            mag_array_per_run[run] = np.arange(mag_params_all_runs[
                run]['mmin'],mag_params_all_runs[run]['mmax'],
                mag_params_all_runs[run]['spacing'])
        self.mag_list_per_config =  mag_array_per_run
        
        # Depths for trellis plots
        self.trellis_depth_per_config = config_file['depth_values']['depth']
        for depth_val in range(0,len(self.trellis_depth_per_config)):
            self.trellis_depth_per_config[depth_val] = float(self.trellis_depth_per_config[
                depth_val])
            
        # Depths for use in other functions
        depth_array_per_run = {}
        for depth in range(0,len(config_file['depth_values']['depth'])):
            depth_array_per_run[depth] = np.full(len(mag_array_per_run[
                str(depth)]),config_file['depth_values']['depth'][depth])
        self.depth_for_non_trellis_functions = depth_array_per_run 
            
        self.aratio = -999
        self.maxR = pd.Series(config_file['maxR_values']['maxR'])
        self.Nstd = pd.Series(config_file['no_std_per_run']['Nstd'])
        
        # Get imts per config
        self.imt_list_per_config = config_file['imt_list']
        
        # Get model labels per config
        self.gmpes_label_per_model_set = config_file['gmpes_label']
        
        # Get models per config 
        config = copy.deepcopy(config_file)
        gmpes_per_set_all={}
        for key1 in config['models']:
            gmpes_per_set = {}
            for key2 in config['models'][key1]:
                if re.search("^\\d+\\-", key2):
                            tmp = re.sub("^\\d+\\-", "", key2)
                            value = f"[{tmp}] "
                else:
                    value = f"[{key2}] "
                if len(config['models'][key1][key2]):
                    config['models'][key1][key2].pop('style', None)
                    value += '\n' + str(toml.dumps(config['models'][key1][
                        key2]))
                gmpes_per_set[key1,key2] = valid.gsim(value) 
            gmpes_per_set_all[key1] = gmpes_per_set
            
        store_list_per_set={}
        self.gmpes_list_store= {}
        for model_set in gmpes_per_set_all:
            gmpe_list=[]
            for gmpe in gmpes_per_set_all[model_set]:
                gmpe_list.append(gmpes_per_set_all[model_set][gmpe])
                store_list_per_set[model_set] = gmpe_list
        self.gmpes_list_store = store_list_per_set        
        
def plot_trellis(self, output_directory):
    """
    Test execution of trellis plotting function for given run configurations
    """
    for t, name in enumerate(self.name_out):
    
        gmpes_list = []
        for gmpe in range(0,len(self.gmpes_list_store[str(t)])):
            gmpes_list.append(str(self.gmpes_list_store[str(t)][gmpe]))
        gmpes_list
        
        mag_list = self.trellis_mag_list_per_config[t]
        
        depth = self.trellis_depth_per_config
        
        imt_list = self.imt_list_per_config[str(t)]['imt_list']
        for imt in range(0,len(imt_list)):
            imt_list[imt] = from_string(str(imt_list[imt]))   

        plot_trellis_util(self.rake[t], self.strike[t], self.dip[t],
                     depth, self.Z1[t], self.Z25[t], self.Vs30[t],
                     self.region[t], imt_list, mag_list,
                     self.maxR[t], gmpes_list, self.aratio, self.Nstd[t],
                     name, output_directory) 
                
def plot_cluster(self, output_directory):
    """
    Plot hierarchical clusters for given run configurations
    """
    for t, name in enumerate(self.name_out):
    
        gmpes_list = []
        for gmpe in range(0,len(self.gmpes_list_store[str(t)])):
            gmpes_list.append(str(self.gmpes_list_store[str(t)][
                gmpe]))
        gmpes_list
        
        if len(gmpes_list) < 2:
            raise ValueError("Cannot perform clustering for a single GMPE.")     
            
        gmpes_label = self.gmpes_label_per_model_set[str(t)][
            'gmpes_label']

        mag_list = self.mag_list_per_config[str(t)]
        
        depth = self.depth_for_non_trellis_functions[t]
        
        imt_list = self.imt_list_per_config[str(t)]['imt_list']
        for imt in range(0,len(imt_list)):
            imt_list[imt] = from_string(str(imt_list[imt]))
        
        mtxs_medians = compute_matrix_gmpes(imt_list, mag_list,
                                            gmpes_list, self.rake[t],
                                            self.strike[t], self.dip[t],
                                            depth, self.Z1[t],
                                            self.Z25[t], self.Vs30[t],
                                            self.region[t], self.maxR[t],
                                            self.aratio)
        
        plot_cluster_util(
            imt_list, gmpes_label, mtxs_medians, os.path.join(
                output_directory,name) + '_Clustering_Vs30_' + str(
                    self.Vs30[t]) +'.png')
                    
def plot_sammons(self, output_directory):
    """
    Plot Sammons Maps for given run configurations
    """
    for t, name in enumerate(self.name_out):
    
        gmpes_list = []
        for gmpe in range(0,len(self.gmpes_list_store[str(t)])):
            gmpes_list.append(str(self.gmpes_list_store[str(t)][
                gmpe]))
        gmpes_list
        
        if len(gmpes_list) < 2:
            raise ValueError("Cannot perform Sammons Mapping for a single GMPE.")
        
        gmpes_label = self.gmpes_label_per_model_set[str(t)][
            'gmpes_label']

        mag_list = self.mag_list_per_config[str(t)]
        
        depth =self.depth_for_non_trellis_functions[t]
        
        imt_list = self.imt_list_per_config[str(t)]['imt_list']
        for imt in range(0,len(imt_list)):
            imt_list[imt] = from_string(str(imt_list[imt]))
        
        mtxs_medians = compute_matrix_gmpes(imt_list, mag_list,
                                            gmpes_list, self.rake[t],
                                            self.strike[t], self.dip[t],
                                            depth, self.Z1[t],
                                            self.Z25[t], self.Vs30[t],
                                            self.region[t], self.maxR[t],
                                            self.aratio)
        
        plot_sammons_util(imt_list, gmpes_label, mtxs_medians, os.path.join(
            output_directory,name) + '_SammonMaps_Vs30_' + str(
                self.Vs30[t])+'.png')
        
def plot_euclidean(self,output_directory):
    """
    Plot Euclidean distance matrix for given run configurations
    """
    for t, name in enumerate(self.name_out):
    
        gmpes_list = []
        for gmpe in range(0,len(self.gmpes_list_store[str(t)])):
            gmpes_list.append(str(self.gmpes_list_store[str(t)][
                gmpe]))
        gmpes_list
        
        if len(gmpes_list) < 2:
            raise ValueError("Cannot perform Euclidean distance matrix plotting for a single GMPE.")
        
        gmpes_label = self.gmpes_label_per_model_set[str(t)][
            'gmpes_label']

        mag_list = self.mag_list_per_config[str(t)]
        
        depth = self.depth_for_non_trellis_functions[t]
        
        imt_list = self.imt_list_per_config[str(t)]['imt_list']
        for imt in range(0,len(imt_list)):
            imt_list[imt] = from_string(str(imt_list[imt]))
        
        mtxs_medians = compute_matrix_gmpes(imt_list, mag_list,
                                            gmpes_list, self.rake[t],
                                            self.strike[t], self.dip[t],
                                            depth, self.Z1[t],
                                            self.Z25[t], self.Vs30[t],
                                            self.region[t], self.maxR[t],
                                            self.aratio)
        
        plot_euclidean_util(imt_list, gmpes_label, mtxs_medians, os.path.join(
            output_directory,name) + '_Euclidean_Vs30_' + str(
                self.Vs30[t]) +'.png')