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
import tempfile
import unittest
import numpy as np
import pandas as pd
import toml

from openquake.hazardlib.imt import from_string
from openquake.smt.comparison import compare_gmpes as comp
from openquake.smt.comparison.utils_gmpes import reformat_att_curves, reformat_spectra
from openquake.smt.comparison.utils_compare_gmpes import (compute_matrix_gmpes,
                                                          plot_cluster_util,
                                                          plot_sammons_util,
                                                          plot_matrix_util)


# Base path
base = os.path.join(os.path.dirname(__file__), "data")

# Defines the target values for each run in the inputted .toml file
TARGET_vs30 = 800
TARGET_DEPTHS = [20, 25, 30]
TARGET_RMIN = 0
TARGET_RMAX = 300
TARGET_NSTD = 2
TARGET_MAGS = [5.0, 6.0, 7.0]
TARGET_MAG_EUC = [5., 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9,
                  6., 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9]
TARGET_IMTS = ['PGA', 'SA(0.1)', 'SA(0.5)', 'SA(1.0)']
TARGET_GMPES = ['[ChiouYoungs2014] \nlt_weight_gmc1 = 0.5',
                '[CampbellBozorgnia2014] \nlt_weight_gmc1 = 0.5',
                '[BooreEtAl2014] \nlt_weight_gmc2_plot_lt_only = 0.5',
                '[KothaEtAl2020] \nlt_weight_gmc2_plot_lt_only = 0.5']
TARGET_BASELINE_GMPE = '[BooreEtAl2014]'
TARGET_TRT = 'active_crustal'
TARGET_ZTOR = -999

# Target for Euclidean distance analysis related matrices
TARGET_EUCL = 4 # 2 GMMs (CY14, CB14), the lt made of them (gmc1) and
                # the second lt (gmc2 - no individual GMMs considered)

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
        self.exp_curves = os.path.join(base,'exp_curves.csv')
        self.exp_spectra = os.path.join(base, 'exp_spectra.csv')
        self.gmc_xml = os.path.join(base, 'comparison_test.xml')

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

        # Check for target vs30
        self.assertEqual(config.vs30, TARGET_vs30)

        # Check for target depths
        np.testing.assert_allclose(config.depth_list, TARGET_DEPTHS)

        # Check for target Rmin
        self.assertEqual(config.minR, TARGET_RMIN)

        # Check for target Rmax
        self.assertEqual(config.maxR, TARGET_RMAX)

        # Check for target Nstd
        self.assertEqual(config.nstd, TARGET_NSTD)

        # Check for target trellis mag
        np.testing.assert_allclose(config.mag_list, TARGET_MAGS)

        # Check for target mag
        np.testing.assert_allclose(config.mags_eucl, TARGET_MAG_EUC)

        # Check for target gmpes
        for gmpe in range(0, len(config.gmpes_list)):
            self.assertEqual(config.gmpes_list[gmpe], TARGET_GMPES[gmpe])

        # Check for target imts
        for imt in range(0, len(config.imt_list)):
            self.assertEqual(str(config.imt_list[imt]), TARGET_IMTS[imt])

        # Check baseline GMPE used to compute ratios
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
        check_imts = []
        for imt in TARGET_IMTS:
            check_imts.append(mtxs_medians[from_string(imt)])
        self.assertEqual(len(check_imts), len(TARGET_IMTS))

        # Check correct number of gmpes
        for imt in TARGET_IMTS:
            self.assertEqual(len(mtxs_medians[from_string(imt)]), len(TARGET_GMPES))

    def test_sammons_and_distance_matrix_functions(self):
        """
        Check expected outputs based on given input parameters for median
        Sammons and Euclidean distance matrix plotting functions
        """
        # Load config
        config = comp.Configurations(self.input_file)

        # Get lts
        lts = 0
        for lt in [config.lt_weights_gmc1, config.lt_weights_gmc2,
                   config.lt_weights_gmc3, config.lt_weights_gmc4]:
            if lt is not None:
                lts += 1

        # Get medians
        mtxs_medians = compute_matrix_gmpes(config, mtxs_type='median')

        # Sammons checks
        coo_per_imt = plot_sammons_util(
            config.imt_list,
            config.gmpe_labels,
            mtxs_medians,
            os.path.join(self.output_directory, 'SammonMaps.png'),
            config.custom_color_flag,
            config.custom_color_list,
            mtxs_type='median')

        # Check Sammons computing correct number of GMPEs and IMTs
        self.assertEqual(list(coo_per_imt.keys()),
                         [from_string(imt) for imt in TARGET_IMTS])
        for imt in TARGET_IMTS:
            imt_sammons = coo_per_imt[from_string(imt)]
            self.assertEqual(len(imt_sammons), TARGET_EUCL)

        # Euclidean checks
        matrix_dist = plot_matrix_util(
            config.imt_list,
            config.gmpe_labels,
            mtxs_medians,
            os.path.join(self.output_directory, 'Euclidean.png'),
            mtxs_type='median')

        # Check correct number of IMTS within matrix_dist
        self.assertEqual(len(matrix_dist), len(TARGET_IMTS))

        # Check correct number of GMPEs within matrix_dist for each IMT
        for imt in range(0, len(matrix_dist)):
            self.assertEqual(len(matrix_dist[imt]), TARGET_EUCL)

        # Check per GMPE that euclidean dist to all other GMPEs is calculated
        for imt in range(0, len(matrix_dist)):
            for gmpe in range(0, len(matrix_dist[imt])):
                self.assertEqual(len(matrix_dist[imt][gmpe]), TARGET_EUCL)

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
        lab = '84th_perc_Clustering_vs30.png'
        Z_matrix = plot_cluster_util(
            config.imt_list,
            config.gmpe_labels,
            mtxs_medians,
            os.path.join(self.output_directory, lab),
            mtxs_type='84th_perc')

        # Check number of cluster arrays matches number of imts per config
        self.assertEqual(len(Z_matrix), len(TARGET_IMTS))

        # Check number of GMPEs matches number of values in each array
        for imt in range(0, len(Z_matrix)):
            for gmpe in range(0, len(Z_matrix[imt])):
                self.assertEqual(len(Z_matrix[imt][gmpe]), len(TARGET_GMPES))

    def test_trellis_and_spectra_functions(self):
        """
        Check trellis and response spectra plotting functions are correctly
        executed. Also checks correct values are returned for the GMPE
        attenuation curves and spectra.
        """
        # Trellis plots
        att_curves = comp.plot_trellis(self.input_file, self.output_directory)
        if not os.path.exists(self.exp_curves):
            # Write to CSV the expected results if missing
            reformat_att_curves(att_curves, self.exp_curves)
        exp_curves = pd.read_csv(self.exp_curves)
        # Same function writing expected can reformat the observed
        obs_curves = reformat_att_curves(att_curves)
        pd.testing.assert_frame_equal(obs_curves, exp_curves, atol=1e-06)

        # Spectra plots
        spectra = comp.plot_spectra(
            self.input_file, self.output_directory, obs_spectra_fname=None)
        if not os.path.exists(self.exp_spectra):
            # Write if doesn't exist
            reformat_spectra(spectra, self.exp_spectra)
        exp_spectra = pd.read_csv(self.exp_spectra, index_col=0)
        # Same function writing expected can reformat the observed
        obs_spectra = reformat_spectra(spectra)
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
        predictions from GMPEs
        """
        # Spectra plots including obs spectra
        comp.plot_spectra(self.input_file_plot_obs_spectra,
                          self.output_directory,
                          self.input_file_obs_spectra_csv)
        
        # Specify target files
        target_file_spectra = (os.path.join(
            self.output_directory, 'ResponseSpectraPlotObserved.png'))
        
        # Check target file created and outputted in expected location
        self.assertTrue(target_file_spectra)

    def test_plot_ratios(self):
        """
        Test execution of plotting ratios (median GMPE attenuation/median
        baseline GMPE attenuation). Correctness of values is not examined.
        """
        # Plot the ratios
        comp.plot_ratios(self.input_file, self.output_directory)

    def test_xml_gmc(self):
        """
        Check that a set of GMCs can be reconstructed correctly from an
        XML for use within the Comparison module. Correctness of values
        is not examined.
        """
        # Add the "gmc_xml" key to the config to override the "models" key
        tmp = toml.load(self.input_file)
        tmp['xml'] = {}
        tmp['xml']['gmc_xml'] = self.gmc_xml
        
        # Test for only ASCR and then all LTs
        for trt in ["Active Shallow Crust", "all"]:

            # Set the TRT
            tmp['xml']['trt'] = trt
            
            # Test for plotting of both individual GMMs and only LTs
            for val in [True, False]:
                
                # Set the plotting option
                tmp['xml']['plot_lt_only'] = val

                # Write back to temp
                tmp_pth = os.path.join(
                    tempfile.mkdtemp(), 'input_with_gmc_xml.toml')
                with open(tmp_pth, 'w', encoding='utf-8') as f:
                    toml.dump(tmp, f)

                # Check the GMCs read from XML work correctly
                comp.plot_trellis(tmp_pth, self.output_directory)
        

    @classmethod
    def tearDownClass(self):
        """
        Remove the test outputs
        """
        shutil.rmtree(self.output_directory)