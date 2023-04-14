"""
Core test suite for the database and residuals construction
"""
import os
import sys
import shutil
import unittest
from openquake.smt.parsers.esm_flatfile_parser import ESMFlatfileParser
import openquake.smt.residuals.gmpe_residuals as res
import openquake.smt.residuals.residual_plotter as rspl

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


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
        Tests basic execution of residuals - not correctness of values
        """
        residuals = res.Residuals(self.gmpe_list, self.imts)
        residuals.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(residuals.residuals)
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

    def test_likelihood_execution_old(self):
        """
        Tests basic execution of residuals - not correctness of values
        """
        lkh = res.Likelihood(self.gmpe_list, self.imts)
        lkh.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(lkh.residuals)
        lkh.get_likelihood_values()

    def test_llh_execution_old(self):
        """
        Tests execution of LLH - not correctness of values
        """
        llh = res.LLH(self.gmpe_list, self.imts)
        llh.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(llh.residuals)
        llh.get_loglikelihood_values(self.imts)

    def test_multivariate_llh_execution_old(self):
        """
        Tests execution of multivariate llh - not correctness of values
        """
        multi_llh = res.MultivariateLLH(self.gmpe_list, self.imts)
        multi_llh.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(multi_llh.residuals)
        multi_llh.get_multivariate_loglikelihood_values()

    def test_edr_execution_old(self):
        """
        Tests execution of EDR - not correctness of values
        """
        edr = res.EDR(self.gmpe_list, self.imts)
        edr.get_residuals(self.database, component="Geometric")
        self._check_residual_dictionary_correctness(edr.residuals)
        edr.get_edr_values()

    def test_single_station_residual_analysis(self):
        """
        Test execution of single station residual analysis functions - not
        correctness of values. Execution of plots is also tested here.
        """
        # Get sites with at least 1 record each
        threshold = 1
        top_sites = res.rank_sites_by_record_count(self.database, threshold)
            
        # Create SingleStationAnalysis object
        ssa1 = res.SingleStationAnalysis(top_sites.keys(), self.gmpe_list,
                                         self.imts)
        
        # Compute the total, inter-event and intra-event residuals for each site
        ssa1.get_site_residuals(self.database)
        
        # Get single station residual statistics per GMPE and per imt
        ssa_csv_output = os.path.join(self.out_location, 'SSA_test.csv')
        ssa1.residual_statistics(True, ssa_csv_output)
        
        # Check num. sites, GMPEs and intensity measures + csv outputted
        self.assertTrue(len(ssa1.site_ids) == len(top_sites))
        self.assertTrue(len(ssa1.gmpe_list) == len(self.gmpe_list))
        self.assertTrue(len(ssa1.imts) == len(self.imts))
        self.assertTrue(ssa_csv_output)
        
        # Check plots outputted for each GMPE and intensity measure
        for gmpe in self.gmpe_list:
            for imt in self.imts:                        
                output_all_res_plt = os.path.join(self.out_location, gmpe +
                                                  imt + 'AllResPerSite.jpg') 
                output_intra_res_comp_plt = os.path.join(self.out_location,
                                                         gmpe + imt +
                                                         'IntraResCompPerSite.jpg') 
                rspl.ResidualWithSite(ssa1, gmpe, imt, output_all_res_plt,
                                      filetype = 'jpg')
                rspl.IntraEventResidualWithSite(ssa1, gmpe, imt,
                                                output_intra_res_comp_plt,
                                                filetype = 'jpg')
                # Check plots outputted
                self.assertTrue(output_all_res_plt)
                self.assertTrue(output_intra_res_comp_plt)
                
    @classmethod
    def tearDownClass(cls):
        """
        Deletes the database
        """
        shutil.rmtree(cls.out_location)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
