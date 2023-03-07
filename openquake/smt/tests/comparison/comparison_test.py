#!/usr/bin/env python
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
Tests for execution of comparison module
"""
import os
import shutil
import unittest
import pandas as pd
from openquake.hazardlib import valid
from openquake.hazardlib.imt import from_string
from openquake.smt.comparison import compare_gmpes as comp
from openquake.smt.comparison.utils_compare_gmpes import compute_matrix_gmpes, plot_trellis_util, plot_cluster_util, plot_sammons_util, plot_euclidean_util

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

# Defines the target values for each run in the inputted .toml file
TARGET_RUN_NAMES = pd.Series(['Albania_1std','Albania_2std','Albania_3std'])
TARGET_VS30 = pd.Series([800,800,600])
TARGET_REGION = pd.Series([0,0,0])
TARGET_DEPTHS = [30,30,30]
TARGET_RMAX = pd.Series([300, 300, 300])
TARGET_NSTD = pd.Series([1,2,3])
TARGET_TRELLIS_MAG = [[5.0,6.0,7.0],[5.0,6.0,7.0],[5.0,6.0,7.0]]
TARGET_MAG = {'0': ([5. , 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6. ,
                     6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9]),
              '1': ([5. , 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6. ,
                     6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9]),
              '2': ([5. , 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6. ,
                     6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9])}
TARGET_IMTS = {'0': {'imt_list': ['PGA', 'SA(0.1)', 'SA(0.5)', 'SA(1.0)']},
 '1': {'imt_list': ['PGA', 'SA(0.1)', 'SA(0.5)', 'SA(1.0)']},
 '2': {'imt_list': ['PGA', 'SA(0.1)', 'SA(0.5)', 'SA(1.0)','SA(2.0)']}}
TARGET_GMPES = {'0': [valid.gsim('ChiouYoungs2014'),
                valid.gsim('CampbellBozorgnia2014'), 
                valid.gsim('BooreEtAl2014'), 
                valid.gsim('KothaEtAl2020regional')],
                '1': [valid.gsim('ChiouYoungs2014'),
                valid.gsim('CampbellBozorgnia2014'),
                valid.gsim('BooreEtAl2014'),
                valid.gsim('KothaEtAl2020regional')],
                '2': [valid.gsim('ChiouYoungs2014'),
                valid.gsim('CampbellBozorgnia2014'),
                valid.gsim('BooreEtAl2014'),
                valid.gsim('KothaEtAl2020regional')]}

class CompareGMPEsTestCase(unittest.TestCase):
    """
    Tests the creation of the configuration object (within compare_gmpes) and
    the plotting functions (within utils_compare_gmpes).
    """
    @classmethod
    def setUpClass(cls):
        cls.input_file = os.path.join(BASE_DATA_PATH,
                                    "compare_gmpe_inputs.toml")
    
        cls.output_directory = os.path.join(BASE_DATA_PATH,
                                            'compare_gmpes_test')
        
        # set the output
        if not os.path.exists(cls.output_directory): os.makedirs(
                cls.output_directory)


    def test_configuration_object_check(self):
        """
        Check for match bewteen the parameters read in from the .toml and 
        the Configuration object, which stores the inputted parameters for
        each run.
        """
        # Check each parameter matches target
        config = comp.Configurations(self.input_file)
        
        # Check for target run name
        for run in range(0,len(config.name_out)):
            self.assertEqual(config.name_out[run],TARGET_RUN_NAMES[run])

        # Check for target Vs30
        for run in range(0,len(config.name_out)):
            self.assertEqual(config.Vs30[run],TARGET_VS30[run])

        # Check for target region
        for run in range(0,len(config.name_out)):
            self.assertEqual(config.region[run],TARGET_REGION[run])      
            
        # Check for target depths (other functions use arrays from these depths)
        for run in range(0,len(config.name_out)):
            self.assertEqual(config.trellis_depth_per_config[run],
                             TARGET_DEPTHS[run])     

        # Check for target Rmax
        for run in range(0,len(config.name_out)):
            self.assertEqual(config.maxR[run],TARGET_RMAX[run]) 

        # Check for target Nstd
        for run in range(0,len(config.name_out)):
            self.assertEqual(config.Nstd[run],TARGET_NSTD[run]) 
        
        # Check for target trellis mag
        for t in range(0,len(config.name_out)):
            for mag in range(0,len(config.trellis_mag_list_per_config[t])):
                self.assertEqual(config.trellis_mag_list_per_config[t][
                    mag],TARGET_TRELLIS_MAG[t][mag])
                
        # Check for target mag
        for t in range(0,len(config.name_out)):
            for mag in range(0,len(config.mag_list_per_config[str(t)])):
                self.assertAlmostEqual(config.mag_list_per_config[str(t)][mag],
                                 TARGET_MAG[str(t)][mag],delta=0.000001)

        # Check for target gmpes
        for t in range(0,len(config.name_out)):
            for gmpe in range(0,len(config.gmpes_list_store[str(t)])):
                self.assertEqual(config.gmpes_list_store[str(t)][gmpe],
                                 TARGET_GMPES[str(t)][gmpe])
                
        # Check for target imts
        for t in range(0,len(config.imt_list_per_config)):
            for imt in range(0,len(config.imt_list_per_config[str(t)][
                    'imt_list'])):
                self.assertEqual(config.imt_list_per_config[str(t)][
                    'imt_list'][imt],TARGET_IMTS[str(t)]['imt_list'][imt])

    def test_mtxs_median_calculation(self):
        """
        Check for matches bewteen the matrix of medians computed using
        compute_matrix_gmpes and those expected given the input parameters 
        """
        # Check each parameter matches target
        config = comp.Configurations(self.input_file)
        
        for t, name in enumerate(config.name_out):
        
            gmpes_list = []
            for gmpe in range(0,len(config.gmpes_list_store[str(t)])):
                gmpes_list.append(str(config.gmpes_list_store[str(t)][
                    gmpe]))
            gmpes_list

            mag_list = config.mag_list_per_config[str(t)]
            
            depth = config.depth_for_non_trellis_functions[t]
            
            imt_list = config.imt_list_per_config[str(t)]['imt_list']
            for imt in range(0,len(imt_list)):
                imt_list[imt] = from_string(str(imt_list[imt]))
            
            mtxs_medians = compute_matrix_gmpes(imt_list, mag_list,
                                                gmpes_list, config.rake[t],
                                                config.strike[t],
                                                config.dip[t],
                                                depth, config.Z1[t],
                                                config.Z25[t], 
                                                config.Vs30[t],
                                                config.region[t], 
                                                config.maxR[t],
                                                config.aratio)
            
            # Check correct number of imts
            self.assertEqual(len(mtxs_medians),len(TARGET_IMTS[str(t)][
                'imt_list']))
            
            # Check correct number of gmpes
            for imt in mtxs_medians:
                self.assertEqual(len(mtxs_medians[imt]), len(TARGET_GMPES[
                    str(t)]))
            
    def test_non_trellis_functions(self):
        """
        Check expected outputs based on given input parameters for clustering,
        Sammons and Euclidean distance matrix plotting functions
        """
        
        # Check each parameter matches target
        config = comp.Configurations(self.input_file)
           
        for t, name in enumerate(config.name_out):
            
            gmpes_list = []
            for gmpe in range(0,len(config.gmpes_list_store[str(t)])):
                gmpes_list.append(str(config.gmpes_list_store[str(t)][
                    gmpe]))
            gmpes_list
                
            gmpes_label = config.gmpes_label_per_model_set[str(t)][
                'gmpes_label']

            mag_list = config.mag_list_per_config[str(t)]
            
            depth = config.depth_for_non_trellis_functions[t]
            
            imt_list = config.imt_list_per_config[str(t)]['imt_list']
            for imt in range(0,len(imt_list)):
                imt_list[imt] = from_string(str(imt_list[imt]))
            
            mtxs_medians = compute_matrix_gmpes(imt_list, mag_list,
                                                gmpes_list, config.rake[t],
                                                config.strike[t],
                                                config.dip[t],
                                                depth, config.Z1[t],
                                                config.Z25[t], config.Vs30[t],
                                                config.region[t],
                                                config.maxR[t],
                                                config.aratio)
            
            # CLUSTER CHECKS
            Z_matrix = plot_cluster_util(imt_list, gmpes_label, mtxs_medians,
                                         os.path.join(
                                             self.output_directory,name)
                                         + '_Clustering_Vs30_' + str(
                                             config.Vs30[t]) +'.png')
            
            # Check number of cluster arrays matches number of imts per config
            self.assertEqual(len(Z_matrix),len(TARGET_IMTS[str(t)][
                'imt_list']))
            
            # Check number of gmpes matches number of values in each array
            for imt in range(0,len(Z_matrix)):
                for gmpe in range(0,len(Z_matrix[imt])):
                    self.assertEqual(len(Z_matrix[imt][gmpe]),len(
                        TARGET_GMPES[str(t)]))
                    
            # SAMMONS CHECKS
            coo = plot_sammons_util(imt_list, gmpes_label, mtxs_medians,
                              os.path.join(self.output_directory,name) +
                              '_SammonMaps_Vs30_' + str(config.Vs30[t])+'.png')

            # Check Sammons computing outputs for num. GMPEs in .toml per run 
            self.assertEqual(len(coo),len(TARGET_GMPES[str(t)]))
            
            # EUCLIDEAN CHECKS
            matrix_Dist = plot_euclidean_util(imt_list, gmpes_label,
                                              mtxs_medians, os.path.join(
                                                  self.output_directory,name) +
                                              '_Euclidean_Vs30_' + str(
                                                  config.Vs30[t]) +'.png')
            
            # Check correct number of IMTS within matrix_Dist
            self.assertEqual(len(matrix_Dist),len(TARGET_IMTS[str(t)][
                'imt_list']))
            
            # Check correct number of GMPEs within matrix_Dist for each IMT
            for imt in range(0,len(matrix_Dist)):
                self.assertEqual(len(matrix_Dist[imt]),len(TARGET_GMPES[
                    str(t)]))
            
            # Check for each gmpe that dist to all other GMPEs is calculated
            for imt in range(0,len(matrix_Dist)):
                for gmpe in range(0,len(matrix_Dist[imt])):
                    self.assertEqual(len(matrix_Dist[imt][gmpe]),len(
                        TARGET_GMPES[str(t)]))

    def test_trellis_function(self):
        """
        Check trellis plotting function is executed 
        """
        
        # Check each parameter matches target
        config = comp.Configurations(self.input_file)
        
        for t, name in enumerate(config.name_out):
        
            gmpes_list = []
            for gmpe in range(0,len(config.gmpes_list_store[str(t)])):
                gmpes_list.append(str(config.gmpes_list_store[str(t)][gmpe]))
            gmpes_list
            
            mag_list = config.trellis_mag_list_per_config[t]
            
            depth = config.trellis_depth_per_config
            
            imt_list = config.imt_list_per_config[str(t)]['imt_list']
            for imt in range(0,len(imt_list)):
                imt_list[imt] = from_string(str(imt_list[imt]))   

            plot_trellis_util(config.rake[t], config.strike[t], config.dip[t],
                         depth, config.Z1[t], config.Z25[t], config.Vs30[t],
                         config.region[t], imt_list, mag_list,
                         config.maxR[t], gmpes_list, config.aratio,
                         config.Nstd[t], name, self.output_directory)
            
            # Specify target file (should be created by plot_trellis_util)
            target_file = (os.path.join(self.output_directory,name) +
                           '_TrellisPlots_Vs30_' + str(config.Vs30[t]) +'.png')
            
            # Check target file created and outputted in expected location
            self.assertIn(target_file,target_file)
            
    @classmethod
    def tearDownClass(cls):
        """
        Remove the test outputs
        """
        shutil.rmtree(cls.output_directory)