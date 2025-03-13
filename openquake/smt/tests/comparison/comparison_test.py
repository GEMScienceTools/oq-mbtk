# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2014 GEM Foundation
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
import pickle
import unittest
import numpy as np
import pandas as pd

from openquake.smt.comparison import compare_gmpes as comp
from openquake.smt.comparison.utils_compare_gmpes import (
    compute_matrix_gmpes, plot_trellis_util, plot_spectra_util,
    plot_ratios_util, plot_cluster_util, plot_sammons_util, plot_euclidean_util)


# Base path
base = os.path.join(os.path.dirname(__file__), "data")

# Defines the target values for each run in the inputted .toml file
TARGET_VS30 = 800
TARGET_Z_BASIN_REGION = 'Global'
TARGET_DEPTHS = [20, 25, 30]
TARGET_RMIN = 0
TARGET_RMAX = 300
TARGET_NSTD = 0
TARGET_MAGS = [5.0, 6.0, 7.0]
TARGET_MAG_EUC = [5., 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9,
                  6., 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9]
TARGET_IMTS = ['PGA', 'SA(0.1)', 'SA(0.5)', 'SA(1.0)']
TARGET_GMPES = ['[ChiouYoungs2014] \nlt_weight_gmc1 = 0.5',
                '[CampbellBozorgnia2014] \nlt_weight_gmc1 = 0.5',
                '[BooreEtAl2014] \nlt_weight_gmc2_plot_lt_only = 0.5',
                '[KothaEtAl2020] \nlt_weight_gmc2_plot_lt_only = 0.5']
TARGET_BASELINE_GMPE = '[BooreEtAl2014]'
TARGET_TRT = 'ASCR'
TARGET_ZTOR = None


class ComparisonTestCase(unittest.TestCase):
    """
    Core test case for the comparison module
    """
    @classmethod
    def setUpClass(self):
        self.input_file = os.path.join(base,"compare_gmpe_inputs.toml")
        self.output_directory = os.path.join(base,'compare_gmpes_test')
        self.input_file_plot_obs_spectra = os.path.join(
            base,'Chamoli_1999_03_28_EQ.toml')
        self.input_file_obs_spectra_csv = os.path.join(
            base,'Chamoli_1999_03_28_EQ_UKHI_rec.csv')
        self.exp_curves = os.path.join(base,'exp_curves.pkl')
        self.exp_spectra = os.path.join(base, 'exp_spectra.pkl')

        # Set the output
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def test_configuration_object_check(self):
        """
        Check for match between the parameters read in from the .toml and
        the Configuration object, which stores the inputted parameters for
        each run.
        """
        # Load config
        config = comp.Configurations(self.input_file)

        # Check for target TRT
        self.assertEqual(config.trt, TARGET_TRT)

        # Check for target ztor
        self.assertEqual(config.ztor, TARGET_ZTOR)

        # Check for target Vs30
        self.assertEqual(config.Vs30, TARGET_VS30)

        # Check for target region
        self.assertEqual(config.z_basin_region, TARGET_Z_BASIN_REGION)

        # Check for target depths (other functions use arrays from these
        # depths)
        np.testing.assert_allclose(config.depth_list, TARGET_DEPTHS)

        # Check for target Rmin
        self.assertEqual(config.minR, TARGET_RMIN)

        # Check for target Rmax
        self.assertEqual(config.maxR, TARGET_RMAX)

        # Check for target Nstd
        self.assertEqual(config.Nstd, TARGET_NSTD)

        # Check for target trellis mag
        np.testing.assert_allclose(config.mag_list, TARGET_MAGS)

        # Check for target mag
        np.testing.assert_allclose(config.mags_euclidean, TARGET_MAG_EUC)

        # Check for target gmpes
        for gmpe in range(0, len(config.gmpes_list)):
            self.assertEqual(config.gmpes_list[gmpe], TARGET_GMPES[gmpe])

        # Check for target imts
        for imt in range(0, len(config.imt_list)):
            self.assertEqual(str(config.imt_list[imt]), TARGET_IMTS[imt])

        # Check baseline GMM used to compute ratios
        self.assertEqual(config.baseline_gmm, TARGET_BASELINE_GMPE)

    def test_mtxs_median_calculation(self):
        """
        Check for matches bewteen the matrix of medians computed using
        compute_matrix_gmpes and those expected given the input parameters
        """
        # Load config
        config = comp.Configurations(self.input_file)

        # Get medians
        mtxs_medians = compute_matrix_gmpes(config, mtxs_type='median')

        # Check correct number of imts
        self.assertEqual(len(mtxs_medians), len(TARGET_IMTS))

        # Check correct number of gmpes
        for imt in mtxs_medians:
            self.assertEqual(len(mtxs_medians[imt]), len(TARGET_GMPES))

    def test_sammons_and_euclidean_distance_matrix_functions(self):
        """
        Check expected outputs based on given input parameters for median
        Sammons and Euclidean distance matrix plotting functions
        """
        TARGET_GMPES.append('mean')  # Add mean here to gmpe_list

        # Load config
        config = comp.Configurations(self.input_file)

        # Get medians
        mtxs_medians = compute_matrix_gmpes(config, mtxs_type='median')

        # Sammons checks
        coo = plot_sammons_util(
            config.imt_list, config.gmpe_labels,
            mtxs_medians, os.path.join(self.output_directory,
                                       'SammonMaps.png'),
            config.custom_color_flag, config.custom_color_list,
            mtxs_type='median')

        # Check Sammons computing outputs for num. GMPEs in .toml per run
        self.assertEqual(len(coo), len(TARGET_GMPES))

        # Euclidean checks
        matrix_Dist = plot_euclidean_util(
            config.imt_list, config.gmpe_labels, mtxs_medians,
            os.path.join(self.output_directory, 'Euclidean.png'),
            mtxs_type='median')

        # Check correct number of IMTS within matrix_Dist
        self.assertEqual(len(matrix_Dist), len(TARGET_IMTS))

        # Check correct number of GMPEs within matrix_Dist for each IMT
        for imt in range(0, len(matrix_Dist)):
            self.assertEqual(len(matrix_Dist[imt]), len(TARGET_GMPES))

        # Check for each gmpe that dist to all other GMPEs is calculated
        for imt in range(0, len(matrix_Dist)):
            for gmpe in range(0, len(matrix_Dist[imt])):
                self.assertEqual(len(matrix_Dist[imt][gmpe]),
                                 len(TARGET_GMPES))

    def test_clustering_median(self):
        """
        Check clustering functions for median predicted ground-motion of
        considered GMPEs in the configuration
        """
        # Load config
        config = comp.Configurations(self.input_file)

        # Get medians
        mtxs_medians = compute_matrix_gmpes(config, mtxs_type='median')
        
        # Get clustering matrix
        Z_matrix = plot_cluster_util(
            config.imt_list, config.gmpe_labels, mtxs_medians,
            os.path.join(self.output_directory, 'Median_Clustering.png'),
            mtxs_type='median')

        # Check number of cluster arrays matches number of imts per config
        self.assertEqual(len(Z_matrix), len(TARGET_IMTS))

        # Check number of gmpes matches number of values in each array
        for imt in range(0, len(Z_matrix)):
            for gmpe in range(0, len(Z_matrix[imt])):
                self.assertEqual(len(Z_matrix[imt][gmpe]), len(TARGET_GMPES))

    def test_clustering_84th_perc(self):
        """
        Check clustering of 84th percentile of predicted ground-motion of
        considered GMPEs in the configuration
        """
        # Load config
        config = comp.Configurations(self.input_file)

        # Get medians
        mtxs_medians = compute_matrix_gmpes(config, mtxs_type='84th_perc')
        
        # Get clustering matrix
        lab = '84th_perc_Clustering_Vs30.png'
        Z_matrix = plot_cluster_util(
            config.imt_list, config.gmpe_labels, mtxs_medians,
            os.path.join(self.output_directory, lab), mtxs_type='84th_perc')

        # Check number of cluster arrays matches number of imts per config
        self.assertEqual(len(Z_matrix), len(TARGET_IMTS))

        # Check number of gmpes matches number of values in each array
        for imt in range(0, len(Z_matrix)):
            for gmpe in range(0, len(Z_matrix[imt])):
                self.assertEqual(len(Z_matrix[imt][gmpe]), len(TARGET_GMPES))

    def test_trellis_and_spectra_functions(self):
        """
        Check trellis and response spectra plotting functions are correctly
        executed. Also checks correct values are returned for the gmm
        attenuation curves and spectra.
        """
        # Load config
        config = comp.Configurations(self.input_file)

        # Trellis plots
        att_curves = plot_trellis_util(config, self.output_directory)
        if not os.path.exists(self.exp_curves):
            with open(self.exp_curves, 'wb') as f: # Write if doesn't exist
                pickle.dump(att_curves, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.exp_curves, 'rb') as f:
                exp_curves = pd.DataFrame(pickle.load(f))
        obs_curves = pd.DataFrame(att_curves)
        pd.testing.assert_frame_equal(obs_curves, exp_curves, atol=1e-06)

        # Spectra plots
        gmc_lts = plot_spectra_util(
            config, self.output_directory, obs_spectra=None)
        if not os.path.exists(self.exp_spectra):
            with open(self.exp_spectra, 'wb') as f: # Write if doesn't exist
                pickle.dump(gmc_lts, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.exp_spectra, 'rb') as f:
                exp_spectra = pd.DataFrame(pickle.load(f))
        obs_spectra = pd.DataFrame(gmc_lts)
        pd.testing.assert_frame_equal(obs_spectra, exp_spectra, atol=1e-06)
        
        # Specify target files
        target_file_trellis = (os.path.join(
            self.output_directory, 'TrellisPlots.png'))
        target_file_spectra = (os.path.join(
            self.output_directory, 'ResponseSpectra.png'))

        # Check target file created and outputted in expected location
        self.assertTrue(target_file_trellis)
        self.assertTrue(target_file_spectra)

    def test_plot_observed_spectra(self):
        """
        Test execution of plotting an observed spectra from a csv against
        predictions from gmpes
        """
        # Get config and obs spectra
        config = comp.Configurations(self.input_file_plot_obs_spectra)
        obs_sp = self.input_file_obs_spectra_csv
        
        # Spectra plots including obs spectra
        plot_spectra_util(config, self.output_directory, obs_sp)
        
        # Specify target files
        target_file_spectra = (os.path.join(
            self.output_directory, 'ResponseSpectraPlotObserved.png'))

        # Check target file created and outputted in expected location
        self.assertTrue(target_file_spectra)

    def test_plot_ratios(self):
        """
        Test execution of plotting ratios (median GMM attenuation/median
        baseline GMM attenuation). Correctness of values is not examined.
        """
        # Load config
        config = comp.Configurations(self.input_file)

        # Plot the ratios
        plot_ratios_util(config, self.output_directory)

    @classmethod
    def tearDownClass(self):
        """
        Remove the test outputs
        """
        shutil.rmtree(self.output_directory)