# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2024 GEM Foundation and G. Weatherill
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
Core test suite for the database and residuals construction
"""
import os
import shutil
import tempfile
import unittest
import pickle
import numpy as np
import pandas as pd

from openquake.smt.residuals.parsers.esm_flatfile_parser import \
    ESMFlatfileParser
import openquake.smt.residuals.gmpe_residuals as res
import openquake.smt.residuals.residual_plotter as rspl
from openquake.smt.residuals.sm_database_selector import \
    rank_sites_by_record_count


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

# Temp file for residuals/ranking metric tables 
tmp_tab = os.path.join(tempfile.mkdtemp(), 'temp_table.csv')
tmp_fig = os.path.join(tempfile.mkdtemp(), 'temp_figure')

EXPECTED_IDS = [
    "EMSC_20040918_0000026_RA_PYAS_0", "EMSC_20040918_0000026_RA_PYAT_0",
    "EMSC_20040918_0000026_RA_PYLI_0", "EMSC_20040918_0000026_RA_PYLL_0",
    "EMSC_20041205_0000033_CH_BNALP_0", "EMSC_20041205_0000033_CH_BOURR_0",
    "EMSC_20041205_0000033_CH_DIX_0", "EMSC_20041205_0000033_CH_EMV_0",
    "EMSC_20041205_0000033_CH_LIENZ_0", "EMSC_20041205_0000033_CH_LLS_0",
    "EMSC_20041205_0000033_CH_MMK_0", "EMSC_20041205_0000033_CH_SENIN_0",
    "EMSC_20041205_0000033_CH_SULZ_0", "EMSC_20041205_0000033_CH_VDL_0",
    "EMSC_20041205_0000033_CH_ZUR_0", "EMSC_20041205_0000033_RA_STBO_0",
    "EMSC_20130103_0000020_HL_SIVA_0", "EMSC_20130103_0000020_HL_ZKR_0",
    "EMSC_20130108_0000044_HL_ALNA_0", "EMSC_20130108_0000044_HL_AMGA_0",
    "EMSC_20130108_0000044_HL_DLFA_0", "EMSC_20130108_0000044_HL_EFSA_0",
    "EMSC_20130108_0000044_HL_KVLA_0", "EMSC_20130108_0000044_HL_LIA_0",
    "EMSC_20130108_0000044_HL_NOAC_0", "EMSC_20130108_0000044_HL_PLG_0",
    "EMSC_20130108_0000044_HL_PRK_0", "EMSC_20130108_0000044_HL_PSRA_0",
    "EMSC_20130108_0000044_HL_SMTH_0", "EMSC_20130108_0000044_HL_TNSA_0",
    "EMSC_20130108_0000044_HL_YDRA_0", "EMSC_20130108_0000044_KO_ENZZ_0",
    "EMSC_20130108_0000044_KO_FOCM_0", "EMSC_20130108_0000044_KO_GMLD_0",
    "EMSC_20130108_0000044_KO_GOKC_0", "EMSC_20130108_0000044_KO_GOMA_0",
    "EMSC_20130108_0000044_KO_GPNR_0", "EMSC_20130108_0000044_KO_KIYI_0",
    "EMSC_20130108_0000044_KO_KRBN_0", "EMSC_20130108_0000044_KO_ORLT_0",
    "EMSC_20130108_0000044_KO_SHAP_0"]


class ResidualsTestCase(unittest.TestCase):
    """
    Core test case for the residuals objects
    """

    @classmethod
    def setUpClass(cls):
        """
        Setup constructs the database from the ESM test data
        """
        ifile = os.path.join(BASE_DATA_PATH, "residual_tests_esm_data.csv")
        cls.out_location = os.path.join(BASE_DATA_PATH, "residual_tests")
        if os.path.exists(cls.out_location):
            shutil.rmtree(cls.out_location)
        parser = ESMFlatfileParser.autobuild("000", "ESM ALL",
                                             cls.out_location, ifile)
        del parser
        cls.database_file = os.path.join(cls.out_location,
                                         "metadatafile.pkl")
        cls.database = None
        with open(cls.database_file, "rb") as f:
            cls.database = pickle.load(f)
        cls.gmpe_list = ["AkkarEtAlRjb2014",  "ChiouYoungs2014"]
        cls.imts = ["PGA", "SA(1.0)"]
        cls.toml = os.path.join(
            BASE_DATA_PATH, 'residuals_from_toml_test.toml')
        cls.exp = exp
        
    def test_correct_build_load(self):
        """
        Verifies that the database has been built and loaded correctly
        """
        self.assertEqual(len(self.database), 41)
        self.assertListEqual([rec.id for rec in self.database],
                             EXPECTED_IDS)

    def _check_residual_dictionary_correctness(self, res_dict):
        """
        Basic check for correctness of the residual dictionary
        """
        for i, gsim in enumerate(res_dict):
            self.assertEqual(gsim, self.gmpe_list[i])
            for j, imt in enumerate(res_dict[gsim]):
                self.assertEqual(imt, self.imts[j])
                if gsim == "AkkarEtAlRjb2014":
                    # For Akkar et al - inter-event residuals should have
                    # 4 elements and the intra-event residuals 41
                    self.assertEqual(
                        len(res_dict[gsim][imt]["Inter event"]), 4)
                elif gsim == "ChiouYoungs2014":
                    # For Chiou & Youngs - inter-event residuals should have
                    # 41 elements and the intra-event residuals 41 too
                    self.assertEqual(
                        len(res_dict[gsim][imt]["Inter event"]), 41)
                else:
                    pass
                self.assertEqual(
                        len(res_dict[gsim][imt]["Intra event"]), 41)
                self.assertEqual(
                        len(res_dict[gsim][imt]["Total"]), 41)

    def test_residuals_execution(self):
        """
        Tests basic execution of residuals and correctness of values
        """
        residuals = res.Residuals(self.gmpe_list, self.imts)
        residuals.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(residuals.residuals)
        residuals.get_residual_statistics()
        obs = pd.DataFrame(residuals.residuals)
        exp = pd.DataFrame(self.exp)
        pd.testing.assert_frame_equal(obs, exp) 

    def test_residuals_execution_from_toml(self):
        """
        Tests basic execution of residuals when specifying gmpe and imts to get
        residuals for from a toml file - not correctness of values
        """
        residuals = res.Residuals.from_toml(self.toml)
        residuals.get_residuals(self.database, component="Geometric")
        residuals.get_residual_statistics()

    def test_likelihood_execution(self):
        """
        Tests basic execution of residuals - not correctness of values
        """
        lkh = res.Residuals(self.gmpe_list, self.imts)
        lkh.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(lkh.residuals)
        lkh.get_likelihood_values()

    def test_llh_execution(self):
        """
        Tests execution of LLH - not correctness of values
        """
        llh = res.Residuals(self.gmpe_list, self.imts)
        llh.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(llh.residuals)
        llh.get_loglikelihood_values(self.imts)

    def test_multivariate_llh_execution(self):
        """
        Tests execution of multivariate llh - not correctness of values
        """
        multi_llh = res.Residuals(self.gmpe_list, self.imts)
        multi_llh.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(multi_llh.residuals)
        multi_llh.get_multivariate_loglikelihood_values()

    def test_edr_execution(self):
        """
        Tests execution of EDR - not correctness of values
        """
        edr = res.Residuals(self.gmpe_list, self.imts)
        edr.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(edr.residuals)
        edr.get_edr_values()
          
    def test_stochastic_area_execution(self):
        """
        Tests execution of stochastic area metric - not correctness of values
        """
        stoch = res.Residuals(self.gmpe_list, self.imts)
        stoch.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(stoch.residuals)
        stoch.get_stochastic_area_wrt_imt()

    def test_multiple_metrics(self):
        """
        Tests the execution running multiple metrics in one call
        """
        residuals = res.Residuals(self.gmpe_list, self.imts)
        residuals.get_residuals(self.database, component="Geometric")
        config = {}
        for key in ["Residuals", "Likelihood", "LLH",
                    "MultivariateLLH", "EDR"]:
            _ = res.GSIM_MODEL_DATA_TESTS[key](residuals, config)

    def test_plot_execution(self):
        """
        Tests execution of gmpe ranking metric plots
        """
        residuals = res.Residuals(self.gmpe_list, self.imts)
        residuals.get_residuals(self.database, component="Geometric")

        # Plots of GMM ranking metrics vs period
        rspl.plot_residual_pdf_with_spectral_period(residuals, tmp_fig)
        rspl.plot_edr_metrics_with_spectral_period(residuals, tmp_fig)
        rspl.plot_loglikelihood_with_spectral_period(residuals, tmp_fig)
        rspl.plot_stochastic_area_with_spectral_period(residuals, tmp_fig)

    def test_table_execution(self):
        """
        Tests execution of table exporting functions
        """
        residuals = res.Residuals(self.gmpe_list, self.imts)
        residuals.get_residuals(self.database, component="Geometric")
        
        # Tables of values
        rspl.pdf_table(residuals, tmp_tab)
        rspl.llh_table(residuals, tmp_tab)
        rspl.edr_table(residuals, tmp_tab)
        rspl.stochastic_area_table(residuals, tmp_tab)
        
        # Tables of weights
        rspl.llh_weights_table(residuals, tmp_tab)
        rspl.edr_weights_table(residuals, tmp_tab)
        rspl.stochastic_area_weights_table(residuals, tmp_tab)
        
    def test_single_station_residual_analysis(self):
        """
        Test execution of single station residual analysis functions - not
        correctness of values. Execution of plots is also tested here.
        """
        # Get sites with at least 1 record each (i.e. all sites in db)
        threshold = 1
        top_sites = rank_sites_by_record_count(self.database, threshold)
            
        # Create SingleStationAnalysis object
        ssa1 = res.SingleStationAnalysis(top_sites.keys(), self.gmpe_list,
                                         self.imts)
        
        # Compute total, inter-event and intra-event residuals for each site
        ssa1.get_site_residuals(self.database)
        
        # Get station residual statistics per GMPE and per imt
        ssa_csv_output = os.path.join(self.out_location, 'SSA_test.csv')
        ssa1.station_residual_statistics(True, ssa_csv_output)
        
        # Check num. sites, GMPEs and intensity measures + csv outputted
        self.assertTrue(len(ssa1.site_ids) == len(top_sites))
        self.assertTrue(len(ssa1.gmpe_list) == len(self.gmpe_list))
        self.assertTrue(len(ssa1.imts) == len(self.imts))
        self.assertTrue(ssa_csv_output)
        
        # Check plots executed for each GMPE and intensity measure
        for gmpe in self.gmpe_list:
            for imt in self.imts:                        
                output_all_res_plt = os.path.join(
                    self.out_location, gmpe + imt + 'AllResPerSite.jpg') 
                output_intra_res_comp_plt = os.path.join(
                    self.out_location, gmpe + imt + 'IntraResCompPerSite.jpg') 
                rspl.ResidualWithSite(
                    ssa1, gmpe, imt, output_all_res_plt, filetype='jpg')
                rspl.IntraEventResidualWithSite(ssa1, gmpe, imt,
                                                output_intra_res_comp_plt,
                                                filetype='jpg')
                # Check plots outputted
                self.assertTrue(output_all_res_plt)
                self.assertTrue(output_intra_res_comp_plt)

    def test_single_station_residual_analysis_from_toml(self):
        """
        Test execution of single station residual analysis using GMPEs and
        imts specified within a toml file. Correctness of values is not tested.
        """
        # Get sites with at least 1 record each (i.e. all sites in db)
        threshold = 1
        top_sites = rank_sites_by_record_count(self.database, threshold)
        
        # Create SingleStationAnalysis object from toml
        ssa1 = res.SingleStationAnalysis.from_toml(top_sites.keys(), self.toml)
        
        # Compute total, inter-event and intra-event residuals for each site
        ssa1.get_site_residuals(self.database)
                
    @classmethod
    def tearDownClass(cls):
        """
        Deletes the database
        """
        shutil.rmtree(cls.out_location)


# Expected residuals
exp = {'AkkarEtAlRjb2014': {'PGA': {'Total': np.array([ 0.09129019, -0.07813266,  0.34160748, -0.14543043,  0.78405242,
       -0.38635982, -1.07660048, -0.07204618,  2.1734739 ,  0.39448275,
        0.09150755, -0.46552581,  1.12788722, -0.31811105,  2.37382625,
        1.20495908, -2.0072862 , -3.50970903, -0.70777005, -3.66567658,
       -2.15148673,  0.17619274, -0.20873352, -0.71933157, -1.13287502,
       -2.3150716 ,  1.082298  ,  0.42332818, -0.0924924 , -2.58651457,
       -1.19630682,  4.28062958,  1.63610396,  2.68515631,  3.22246825,
        1.2883538 ,  2.24626416,  3.40018654,  2.04583726,  3.40806806,
        1.79685995]), 'Inter event': np.array([ 0.0596577 ,  0.78359362, -2.1843895 ,  1.00508837]), 'Intra event': np.array([ 0.07115315, -0.12340727,  0.35861046, -0.20069013,  0.45797741,
       -0.88609098, -1.67874388, -0.52514207,  2.05354983,  0.01060659,
       -0.33732158, -0.97700298,  0.85282759, -0.80771602,  2.28362882,
        0.94133475, -1.07183433, -2.79717432, -1.38024233, -4.77701874,
       -3.03816584, -0.3651244 , -0.80716286, -1.39351926, -1.86842088,
       -3.22602208,  0.67542131, -0.08132104, -0.67367485, -3.53773953,
       -1.94126418,  4.34829505,  1.31139644,  2.51609856,  3.13313243,
        0.91204998,  2.01208721,  3.33721911,  1.7819226 ,  3.34627002,
        1.49600408])}, 'SA(1.0)': {'Total': np.array([ 0.97914815,  1.47509703,  1.79933545,  1.92947892,  1.21249336,
        0.58341771, -0.09504146,  0.42319944,  1.45542195, -0.75249643,
        0.5702202 ,  0.75204581,  1.80455022, -0.75292723,  1.15287599,
       -0.52550586, -0.97115965, -1.45519997, -0.02714449, -3.04429726,
       -0.6131663 ,  0.59275865, -0.92333935, -0.95797589, -0.39521711,
       -1.8081914 ,  0.70790908, -0.16141704,  0.67614808, -2.04728085,
        0.41571142,  3.95568072,  1.22705777,  2.4315021 ,  3.63862258,
        1.93424677,  2.12710294,  2.76001048,  2.20318075,  3.53460709,
        2.07340155]), 'Inter event': np.array([ 1.76774898,  0.77540147, -0.97326144,  1.40313178]), 'Intra event': np.array([ 1.05398208e-01,  6.78968771e-01,  1.05395421e+00,  1.20446662e+00,
        9.51782408e-01,  2.24249224e-01, -5.60396572e-01,  3.89549564e-02,
        1.23273211e+00, -1.32075076e+00,  2.08986149e-01,  4.19269548e-01,
        1.63650293e+00, -1.32124899e+00,  8.82834237e-01, -1.05823358e+00,
       -5.57727767e-01, -1.11752594e+00, -8.46561411e-01, -4.33593311e+00,
       -1.52430234e+00, -1.29636341e-01, -1.88302102e+00, -1.92307857e+00,
       -1.27224161e+00, -2.90636254e+00,  3.53645493e-03, -1.00184915e+00,
       -3.31955086e-02, -3.18287222e+00, -3.34393484e-01,  3.75962151e+00,
        6.03937848e-01,  1.99689150e+00,  3.39294014e+00,  1.42181002e+00,
        1.64485039e+00,  2.37681520e+00,  1.73283525e+00,  3.27264504e+00,
        1.58274412e+00])}}, 'ChiouYoungs2014': {'PGA': {'Total': np.array([ 1.65375303,  0.83432797,  2.19086657,  2.66635996,  0.83928786,
       -0.5719685 , -0.14157983,  0.90627583,  2.05653901,  0.52406656,
        0.93742803,  0.01987959,  0.80439518,  0.33916661,  1.98884822,
        0.78088407, -3.63271379, -4.6657945 , -1.34206861, -3.13906814,
       -1.54089755, -0.39632959, -0.57601495, -1.38245942, -1.04450946,
       -2.74267957,  0.5065491 , -0.1773138 , -0.63464881, -2.71226556,
       -0.42427308,  3.90789901,  1.22756611,  2.8445371 ,  2.86001893,
        1.42437903,  1.81927845,  4.38914723,  1.70059585,  4.26869055,
        2.90161576]), 'Inter event': np.array([ 2.11257686,  2.11257669,  2.11257669,  2.11257671,  1.1107619 ,
        1.11082654,  1.1107619 ,  1.1107619 ,  1.11078006,  1.11076878,
        1.1107619 ,  1.11077509,  1.11089216,  1.1107631 ,  1.11080335,
        1.11538111, -3.39723346, -3.38500401,  0.91765884,  0.91331207,
        0.91305197,  0.91596197,  0.91335584,  0.92306118,  0.91318497,
        0.91377808,  0.91475813,  0.9152861 ,  0.91307085,  0.91367375,
        0.91306277,  0.92041688,  0.91463776,  0.91311973,  0.91367031,
        0.91344305,  0.91458491,  0.91318251,  0.91381158,  0.91308103,
        0.91306737]), 'Intra event': np.array([ 0.63036249, -0.33575335,  1.26362844,  1.82424312,  0.30759455,
       -1.34329232, -0.839815  ,  0.38595643,  1.7315124 , -0.06115078,
        0.42239798, -0.65094512,  0.26672462, -0.27744286,  1.65231494,
        0.23742455, -2.19059541, -3.4034446 , -2.05629657, -4.1190777 ,
       -2.28421285, -0.97074309, -1.17645768, -2.10302467, -1.71430936,
       -3.66392486,  0.06591776, -0.71920736, -1.24372112, -3.62902203,
       -1.00218035,  3.96532611,  0.89361267,  2.75080681,  2.76818813,
        1.120128  ,  1.57286566,  4.52415147,  1.43704673,  4.38594758,
        2.81637798])}, 'SA(1.0)': {'Total': np.array([ 1.07440353,  1.74133924,  1.96498514,  2.25280764,  1.42993817,
        0.69425219,  0.24611266,  0.77173846,  1.4807836 , -0.64522466,
        0.89022263,  0.82594789,  1.95498402, -0.57136565,  1.24953937,
       -0.36297775, -0.1787276 , -0.61005904, -0.35813172, -3.02100154,
       -0.29314596,  0.25813036, -1.18516298, -1.44139226, -0.44870543,
       -2.09911503,  0.43142256, -0.48865153,  0.55926413, -2.24382343,
        0.67443871,  3.80276376,  1.00148654,  2.45808147,  3.5299444 ,
        1.94219277,  1.91347743,  3.06916792,  2.05019414,  3.85259704,
        2.45551157]), 'Inter event': np.array([ 2.0293143 ,  2.02931425,  2.02931425,  2.02931427,  1.00911991,
        1.00912525,  1.00911991,  1.00911991,  1.00912212,  1.00912076,
        1.00911991,  1.00912254,  1.00912862,  1.00912011,  1.00912315,
        1.00965397, -0.3341067 , -0.3340657 ,  1.23918152,  1.23593951,
        1.23554763,  1.23652684,  1.2357129 ,  1.23962707,  1.23565088,
        1.23609042,  1.23619479,  1.236796  ,  1.23555207,  1.23616856,
        1.23555675,  1.24034923,  1.2365343 ,  1.23558559,  1.23572738,
        1.23588733,  1.23603575,  1.23570687,  1.23596062,  1.23557148,
        1.23556238]), 'Intra event': np.array([-0.07162288,  0.73253333,  1.00219386,  1.34923516,  1.04642626,
        0.16336761, -0.37453931,  0.25637814,  1.10745554, -1.44442659,
        0.39859666,  0.32144535,  1.67664158, -1.35577224,  0.82988874,
       -1.10578417,  0.0072677 , -0.51047859, -1.16256166, -4.26945949,
       -1.08571767, -0.44264827, -2.12684534, -2.426464  , -1.26730286,
       -3.19355275, -0.24029179, -1.31422172, -0.09084736, -3.36244074,
        0.04357424,  3.69105111,  0.42485429,  2.12529487,  3.37618732,
        1.52302352,  1.48942939,  2.8384258 ,  1.64902954,  3.75287824,
        2.1223096 ])}}}