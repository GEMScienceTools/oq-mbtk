#!/usr/bin/env python
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
Tests for execution of comparison module
"""
import os
import shutil
import unittest
from openquake.hazardlib import valid
#from openquake.hazardlib.imt import from_string
from openquake.smt.comparison import compare_gmpes as comp
from openquake.smt.comparison.utils_compare_gmpes import compute_matrix_gmpes, plot_trellis_util, plot_cluster_util, plot_sammons_util, plot_euclidean_util

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

# Defines the target values for each run in the inputted .toml file
TARGET_CONFIG_NAME = 'Albania_2std'
TARGET_VS30 = 800
TARGET_REGION = 0
TARGET_DEPTHS = [20,25,30]
TARGET_RMAX = 300
TARGET_NSTD = 2
TARGET_TRELLIS_MAG = [5.0,6.0,7.0]
TARGET_MAG = [5. , 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6. ,
                     6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9]
TARGET_IMTS = ['PGA', 'SA(0.1)', 'SA(0.5)', 'SA(1.0)', 'SA(2.0)']
TARGET_GMPES = [valid.gsim('ChiouYoungs2014'),
                valid.gsim('CampbellBozorgnia2014'), 
                valid.gsim('BooreEtAl2014'), 
                valid.gsim('KothaEtAl2020regional')]
                
class ComparisonTestCase(unittest.TestCase):
    """
    Core test case for the comparison module
    """
    @classmethod 
    def setUpClass(self):
        self.input_file = os.path.join(BASE_DATA_PATH,
                                "compare_gmpe_inputs.toml")
    
        self.output_directory = os.path.join(BASE_DATA_PATH,
                                        'compare_gmpes_test')
    
        # set the output
        if not os.path.exists(self.output_directory): os.makedirs(
                self.output_directory)
    
    
    def test_configuration_object_check(self):
        """
        Check for match bewteen the parameters read in from the .toml and 
        the Configuration object, which stores the inputted parameters for
        each run.
        """
        # Check each parameter matches target
        config = comp.Configurations(self.input_file)
        
        # Check for target run name
        self.assertEqual(config.name_out,TARGET_CONFIG_NAME)
        
        # Check for target Vs30
        self.assertEqual(config.Vs30,TARGET_VS30)
        
        # Check for target region
        self.assertEqual(config.region,TARGET_REGION)      
            
        # Check for target depths (other functions use arrays from these depths)
        self.assertEqual(config.trellis_depth, TARGET_DEPTHS)     
        
        # Check for target Rmax
        self.assertEqual(config.maxR,TARGET_RMAX) 
        
        # Check for target Nstd
        self.assertEqual(config.Nstd,TARGET_NSTD) 
        
        # Check for target trellis mag
        for mag in range(0,len(config.trellis_mag_list)):
            self.assertEqual(config.trellis_mag_list[mag],TARGET_TRELLIS_MAG[
                mag])
                
        # Check for target mag
        for mag in range(0,len(config.mag_list)):
            self.assertAlmostEqual(config.mag_list[mag], TARGET_MAG[mag],
                                   delta=0.000001)
        # Check for target gmpes
        for gmpe in range(0,len(config.gmpes_list)):
            self.assertEqual(config.gmpes_list[gmpe],TARGET_GMPES[gmpe])
                
        # Check for target imts
        for imt in range(0,len(config.imt_list)):
            self.assertEqual(str(config.imt_list[imt]),TARGET_IMTS[imt])
    
    def test_mtxs_median_calculation(self):
        """
        Check for matches bewteen the matrix of medians computed using
        compute_matrix_gmpes and those expected given the input parameters 
        """
        # Check each parameter matches target
        config = comp.Configurations(self.input_file)
        
        mtxs_medians = compute_matrix_gmpes(config.imt_list, config.mag_list,
                                                config.gmpes_list, config.rake,
                                                config.strike, config.dip, 
                                                config.depth_for_non_trellis_functions,
                                                config.Z1, config.Z25,
                                                config.Vs30, config.region,
                                                config.maxR, config.aratio)
        
        # Check correct number of imts
        self.assertEqual(len(mtxs_medians),len(TARGET_IMTS))
        
        # Check correct number of gmpes
        for imt in mtxs_medians:
            self.assertEqual(len(mtxs_medians[imt]), len(TARGET_GMPES))
        
    def test_non_trellis_functions(self):
        """
        Check expected outputs based on given input parameters for clustering,
        Sammons and Euclidean distance matrix plotting functions
        """
        
        # Check each parameter matches target
        config = comp.Configurations(self.input_file)
           
            
        mtxs_medians = compute_matrix_gmpes(config.imt_list, config.mag_list,
                                                config.gmpes_list, config.rake,
                                                config.strike, config.dip, 
                                                config.depth_for_non_trellis_functions,
                                                config.Z1, config.Z25,
                                                config.Vs30, config.region,
                                                config.maxR, config.aratio)
        
            
        # CLUSTER CHECKS
        Z_matrix = plot_cluster_util(config.imt_list, config.gmpe_labels,
                                     mtxs_medians, os.path.join(
                                         self.output_directory,config.name_out)
                                         + '_Clustering_Vs30_' +
                                         str(config.Vs30) +'.png')
            
        # Check number of cluster arrays matches number of imts per config
        self.assertEqual(len(Z_matrix),len(TARGET_IMTS))
            
        # Check number of gmpes matches number of values in each array
        for imt in range(0,len(Z_matrix)):
            for gmpe in range(0,len(Z_matrix[imt])):
                self.assertEqual(len(Z_matrix[imt][gmpe]),len(TARGET_GMPES))
                    
        # SAMMONS CHECKS
        coo = plot_sammons_util(config.imt_list, config.gmpe_labels,
                                mtxs_medians, os.path.join(
                                    self.output_directory,config.name_out) +
                          '_SammonMaps_Vs30_' + str(config.Vs30)+'.png')
        
        # Check Sammons computing outputs for num. GMPEs in .toml per run 
        self.assertEqual(len(coo),len(TARGET_GMPES))
            
        # EUCLIDEAN CHECKS
        matrix_Dist = plot_euclidean_util(config.imt_list, config.gmpe_labels,
                                          mtxs_medians, os.path.join(
                                              self.output_directory,
                                              config.name_out) +
                                          '_Euclidean_Vs30_' + str(
                                              config.Vs30) +'.png')
            
        # Check correct number of IMTS within matrix_Dist
        self.assertEqual(len(matrix_Dist),len(TARGET_IMTS))
            
        # Check correct number of GMPEs within matrix_Dist for each IMT
        for imt in range(0,len(matrix_Dist)):
            self.assertEqual(len(matrix_Dist[imt]),len(TARGET_GMPES))
        
        # Check for each gmpe that dist to all other GMPEs is calculated
        for imt in range(0,len(matrix_Dist)):
            for gmpe in range(0,len(matrix_Dist[imt])):
                self.assertEqual(len(matrix_Dist[imt][gmpe]),len(TARGET_GMPES))
    
    def test_trellis_function(self):
        """
        Check trellis plotting function is executed 
        """
        
        # Check each parameter matches target
        config = comp.Configurations(self.input_file)
        
        plot_trellis_util(config.rake, config.strike, config.dip,
                     config.trellis_depth, config.Z1, config.Z25, config.Vs30,
                     config.region, config.imt_list, config.trellis_mag_list,
                     config.maxR, config.gmpes_list, config.aratio,
                     config.Nstd, config.name_out, self.output_directory)
            
        # Specify target file (should be created by plot_trellis_util)
        target_file = (os.path.join(self.output_directory,config.name_out) +
                       '_TrellisPlots_Vs30_' + str(config.Vs30) +'.png')
        
        # Check target file created and outputted in expected location
        self.assertIn(target_file,target_file)
        
    @classmethod
    def tearDownClass(self):
        """
        Remove the test outputs
        """
        shutil.rmtree(self.output_directory)