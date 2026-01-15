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
Core tests for the SMT's Comparison module.
"""
import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
import toml

from openquake.smt.comparison import compare_gmpes as comp
from openquake.smt.comparison.utils_gmpes import reformat_att_curves, reformat_spectra
from openquake.smt.comparison.utils_compare_gmpes import (compute_matrix_gmpes,
                                                          plot_cluster_util,
                                                          plot_sammons_util,
                                                          plot_matrix_util)


# Base path
BASE = os.path.join(os.path.dirname(__file__), "data")

# Defines the target values for each run in the inputted .toml file
TARGET_VS30 = 800
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
TARGET_EUCL = 4 # 2 GMMs (CY14, CB14), the lt made of them (gmc1) and
                # the second lt (gmc2 - no individual GMMs considered)


class ComparisonTestCase(unittest.TestCase):
    """
    Core test case for the comparison module.
    """
    @classmethod
    def setUpClass(self):
        self.input_file = os.path.join(BASE, "comparison_test.toml")
        self.outdir = os.path.join(BASE, 'compare_gmpes_test')
        self.input_file_plot_obs_spectra = os.path.join(
            BASE, 'Chamoli_1999_03_28_EQ.toml')
        self.input_file_obs_spectra_csv = os.path.join(
            BASE, 'Chamoli_1999_03_28_EQ_UKHI_rec.csv')
        self.exp_curves = os.path.join(BASE, 'exp_curves.csv')
        self.exp_spectra = os.path.join(BASE, 'exp_spectra.csv')
        self.rup_xml = os.path.join(BASE, 'rup.xml')
        self.rup_csv = os.path.join(BASE, 'rup.csv')
        self.gmc_xml = os.path.join(BASE, 'gmm_lt.xml')

        # Set the output
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

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
        self.assertEqual(config.vs30, TARGET_VS30)

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

    def test_sammons(self):
        """
        Check execution of Sammon maps plotting functions when considering
        the median, 16th percentile and 84th percentiles of the distributions
        of predicted ground-motion from the considered GMPEs.
        """
        # Load config
        config = comp.Configurations(self.input_file)

        # For each percentile test the clustering plots
        for perc in ["16th_perc", "median", "84th_perc"]:
        
            # Get the matrix of predictions for the GMPEs
            mtxs = compute_matrix_gmpes(config, mtxs_type=perc)

            # Sammons checks
            coo_per_imt = plot_sammons_util(
            config.imt_list,
            config.gmpe_labels,
            mtxs,
            os.path.join(self.outdir, 'SammonMaps.png'),
            config.custom_color_flag,
            config.custom_color_list,
            mtxs_type='median')

            # Check Sammons computing correct number of IMTs
            self.assertEqual(list(coo_per_imt.keys()), [imt for imt in TARGET_IMTS])
            
            # Check for each IMT we have correct number of values
            for imt in config.imt_list:
                self.assertEqual(len(coo_per_imt[imt]), TARGET_EUCL)

    def test_clustering(self):
        """
        Check execution of dendrogram plotting functions when considering the
        median, 16th percentile and 84th percentiles of the distributions of
        predicted ground-motion from the considered GMPEs.
        """
        # Load config
        config = comp.Configurations(self.input_file)

        # For each percentile test the clustering plots
        for perc in ["16th_perc", "median", "84th_perc"]:

            # Get the matrix of predictions for the GMPEs
            mtxs = compute_matrix_gmpes(config, mtxs_type=perc)
        
            # Get clustering matrix
            z_matrix = plot_cluster_util(config.imt_list,
                                         config.gmpe_labels,
                                         mtxs,
                                         os.path.join(self.outdir, f'{[perc]}_Clustering.png'),
                                         mtxs_type=perc)

            # Check number of cluster arrays matches number of imts
            self.assertEqual(len(z_matrix), len(TARGET_IMTS))

            # Check number of gmpes matches number of values in each IMT's array
            for imt in config.imt_list:
                for gmpe in range(0, len(z_matrix[imt])):
                    self.assertEqual(len(z_matrix[imt][gmpe]), len(TARGET_GMPES))

    def test_distance_matrix(self):
        """
        Check execution of Euclidean distance matrix plotting functions when
        considering the median, 16th percentile and 84th percentiles of the
        distributions of predicted ground-motion from the considered GMPEs.
        """
        # Load config
        config = comp.Configurations(self.input_file)

        # For each percentile test the matrix plots
        for perc in ["16th_perc", "median", "84th_perc"]:

            # Get the matrix of predictions for the GMPEs
            mtxs = compute_matrix_gmpes(config, mtxs_type=perc)
            
            # Euclidean checks
            matrix_dist = plot_matrix_util(
                config.imt_list,
                config.gmpe_labels,
                mtxs,
                os.path.join(self.outdir, 'Euclidean.png'),
                mtxs_type=perc
                )

            # Check correct number of IMTS within matrix_dist
            self.assertEqual(len(matrix_dist), len(TARGET_IMTS))

            # Check correct number of GMPEs within matrix_dist for each IMT
            for imt in config.imt_list:
                self.assertEqual(len(matrix_dist[imt]), TARGET_EUCL)

    def test_trellis_and_spectra_functions(self):
        """
        Check trellis and response spectra plotting functions are correctly
        executed. Also checks correct values are returned for the GMPE
        attenuation curves and spectra.
        """
        # Trellis plots
        att_curves = comp.plot_trellis(self.input_file, self.outdir)
        if not os.path.exists(self.exp_curves):
            # Write to CSV the expected results if missing
            reformat_att_curves(att_curves, self.exp_curves)
        exp_curves = pd.read_csv(self.exp_curves)
        # Same function writing expected can reformat the observed
        obs_curves = reformat_att_curves(att_curves)
        pd.testing.assert_frame_equal(obs_curves, exp_curves, atol=1e-06)

        # Spectra plots
        spectra = comp.plot_spectra(
            self.input_file, self.outdir, obs_spectra_fname=None)
        if not os.path.exists(self.exp_spectra):
            # Write if doesn't exist
            reformat_spectra(spectra, self.exp_spectra)
        exp_spectra = pd.read_csv(self.exp_spectra, index_col=0)
        obs_spectra = reformat_spectra(spectra) 
        # Same function writing expected can reformat the observed
        pd.testing.assert_frame_equal(obs_spectra, exp_spectra, atol=1e-06)
        
        # Specify target files
        target_file_trellis = (os.path.join(
            self.outdir, 'TrellisPlots.png'))
        target_file_spectra = (os.path.join(
            self.outdir, 'ResponseSpectra.png'))

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
                          self.outdir,
                          self.input_file_obs_spectra_csv)
        
        # Specify target files
        target_file_spectra = (os.path.join(
            self.outdir, 'ResponseSpectraPlotObserved.png'))
        
        # Check target file created and outputted in expected location
        self.assertTrue(target_file_spectra)

    def test_plot_ratios(self):
        """
        Test execution of plotting ratios (median GMPE attenuation/median
        baseline GMPE attenuation). Correctness of values is not examined.
        """
        # Plot the ratios
        comp.plot_ratios(self.input_file, self.outdir)

    def test_rup_file(self):
        """
        Check that the provision of an OQ rupture in XML or CSV format is
        usable within the Comparison module. Correctness of values is not
        examined.
        """
        # Add the "rup_file" key to the config to override source params key
        tmp = toml.load(self.input_file)
        tmp['rup_file'] = {}

        # For XML and CSV formats
        for file in [self.rup_xml, self.rup_csv]:

            # Set the file
            tmp['rup_file']['fname'] = file

            # Write back to temp
            tmp_pth = os.path.join(
                tempfile.mkdtemp(), 'input_with_gmc_xml.toml')
            with open(tmp_pth, 'w', encoding='utf-8') as f:
                toml.dump(tmp, f)

            # Check the rup read from file works correctly
            comp.plot_trellis(tmp_pth, self.outdir)

    def test_xml_gmc(self):
        """
        Check that a set of GMCs can be reconstructed correctly from an
        XML for use within the Comparison module. Correctness of values
        is not examined.
        """
        # Add the "gmc_xml" key to the config to override the "models" key
        tmp = toml.load(self.input_file)
        tmp['gmc_xml'] = {}
        tmp['gmc_xml']['fname'] = self.gmc_xml
        
        # Test for only ASCR and then all LTs
        for trt in ["Active Shallow Crust", "all"]:

            # Set the TRT
            tmp['gmc_xml']['trt'] = trt
            
            # Test for plotting of both individual GMMs and only LTs
            for val in [True, False]:
                
                # Set the plotting option
                tmp['gmc_xml']['plot_lt_only'] = val

                # Write back to temp
                tmp_pth = os.path.join(
                    tempfile.mkdtemp(), 'input_with_gmc_xml.toml')
                with open(tmp_pth, 'w', encoding='utf-8') as f:
                    toml.dump(tmp, f)

                # Check the GMCs read from XML work correctly
                comp.plot_trellis(tmp_pth, self.outdir)

    @classmethod
    def tearDownClass(self):
        """
        Remove the test outputs.
        """
        shutil.rmtree(self.outdir)