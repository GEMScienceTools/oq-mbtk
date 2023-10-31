"""
Test for functions added to gmpe_residuals (get_edr_values_wrt_spectral_period,
_get_edr_gmpe_information_wrt_spectral_period and _get_edr_wrt_spectral_period)
and residual_plotter (PlotLoglikelihoodWithSpectralPeriod,
PlotLLHModelWeightsWithSpectralPeriod, PlotEDRWithSpectralPeriod,
LoglikelihoodTable, LLHWeightsTable and EDRTable) to output loglikelihood values
and sample loglikelihood based GMPE weightings (Scherbaum et al., 2009) and EDR
metrics (Kale and Akkar, 2013) w.r.t. spectral period (rather than aggregated
over all intensity measures as before).
"""

import os
import shutil
import pickle
import tempfile
import unittest
import numpy as np
import pandas as pd
from openquake.smt.parsers.esm_flatfile_parser import ESMFlatfileParser
import openquake.smt.residuals.gmpe_residuals as res
import openquake.smt.residuals.residual_plotter as rspl


DATA = os.path.dirname(__file__)


class gmpe_ranking_metrics_wrt_imt_test(unittest.TestCase):
    """
    Test for the additional ranking metric functions (plotting of LLH and EDR
    vs spectral period + tables of EDR and LLH scores + normalised model 
    weights).
    """
    """
    Parse the test flatfile and get the residuals
    """
    
    def setUp(self):
        """
        Parse the test flatfile, create the metadata and get the residuals/
        """
        # Parse test flatfile
        input_fi = os.path.join(DATA, 'data','Ranking_Metrics_Test_Flatfile.csv')
        output_database = os.path.join(DATA, 'data', 'metadata')
        if os.path.exists(output_database):
            shutil.rmtree(output_database)
        parser = ESMFlatfileParser.autobuild(
            "000", "ranking metrics wrt period test", output_database, input_fi)       
        
        # Create the metadata
        metadata_directory = os.path.join(DATA, 'data', 'metadata')
        metadata_file = 'metadatafile.pkl'
        metadata = os.path.join(metadata_directory, metadata_file)
        sm_database = pickle.load(open(metadata,"rb"))
    
        gmpe_list = ['ChiouYoungs2014','CampbellBozorgnia2014',
                     'BooreEtAl2014','AbrahamsonEtAl2014']
        imts = ['PGA', 'SA(1.0)']
        
        # Get the residuals
        self.residuals = res.Residuals(gmpe_list, imts)
        self.residuals.get_residuals(sm_database)
    
        # Set filename
        self.output_directory = tempfile.mkdtemp()
        self.filename = os.path.join(self.output_directory, 'test.csv')

    def llh_test(self):
        """
        Check LLH and LLH-based weights per imt when aggregated over said imts
        match those of the original LLH function.
        """
        # LLH and LLH-based weights aggregated over all imts (original function)
        original_metrics = self.residuals.get_loglikelihood_values(self.residuals.imts)
        original_llh = original_metrics[0]
        avg_model_weights_original = original_metrics[1]
        avg_model_weights_original = np.array(
            pd.Series(avg_model_weights_original))
    
        avg_llh_orig = {}
        for gmpe in self.residuals.gmpe_list:
            avg_llh_orig[gmpe] = original_llh[gmpe]['All']
        avg_llh_orig = pd.Series(avg_llh_orig)
    
        # Get LLH per imt
        llh_per_imt = self.rspl.loglikelihood_table(self.residuals, self.filename) 
        avg_llh_over_imts = llh_per_imt.loc['Avg over all periods']
        avg_llh_new = np.array(pd.Series(avg_llh_over_imts))
    
        # Get model weights per imt
        model_weights_per_imt = rspl.llh_weights_table(self.residuals,
                                                       self.filename)
    
        # Evaluate equivalencies of values computed from w.r.t. imt function
        # outputs and original function outputs
        check_llh = avg_llh_new/avg_llh_orig
        check_wts = (model_weights_per_imt.loc['Avg over all periods']/
                     avg_model_weights_original)
        
        # Check values
        checks = np.array(check_llh, check_wts)
        if np.all(checks) < 1.005 and np.all(checks) > 0.995: # %5 relative tol
            pass
        else:
            raise ValueError('Too large a relative difference between LLH functions')

    def edr_test(self):
        """
        Check EDR and EDR-based weights per imt when aggregated over said imts 
        match those of the original LLH function.
        """
        # Get median correction factor kappa (aggregated over all imts) per gmpe 
        kappa_ratio = []
        for gmpe in self.residuals.gmpe_list:
           
            # Get obs, exp and std per imt
           (obs_wrt_imt, expected_wrt_imt, stddev_wrt_imt
            ) = self.residuals._get_edr_gmpe_information_wrt_imt(gmpe)
           
           # Get obs, exp and std aggregated over all imts ('original' function)
           obs, expected, stddev = self.residuals._get_edr_gmpe_information(gmpe)
           
           # Compute 'original' kappa ratio
           mu_a = np.mean(obs)
           mu_y = np.mean(expected)
           b_1 = np.sum((obs - mu_a) * (expected - mu_y)) /\
    np.sum((obs - mu_a) ** 2.)
           b_0 = mu_y - b_1 * mu_a
           y_c = expected - ((b_0 + b_1 * obs) - obs)
           de_orig = np.sum((obs - expected) ** 2.)
           de_corr = np.sum((obs - y_c) ** 2.)
           original_kappa = de_orig / de_corr
             
           # Compute 'new' kappa from reassembled subsets
           all_new_function_obs=pd.DataFrame()
           all_new_function_expected=pd.DataFrame()
           for imtx in self.residuals.imts:
               all_new_function_obs[imtx]=obs_wrt_imt[imtx]
               all_new_function_expected[imtx]=expected_wrt_imt[imtx]
           all_new_function_expected
             
           all_new_function_obs_1 = np.array(all_new_function_obs)
           all_new_function_expected_1 = np.array(all_new_function_expected)
             
           mu_a = np.mean(all_new_function_obs_1)
           mu_y = np.mean(all_new_function_expected_1)
           b_1 = np.sum((all_new_function_obs_1 - mu_a) * (
               all_new_function_expected_1 - mu_y)) /\
                     np.sum((all_new_function_obs_1 - mu_a) ** 2.)
           b_0 = mu_y - b_1 * mu_a
           y_c = all_new_function_expected_1 - ((b_0 + b_1 * all_new_function_obs_1)
                                                - all_new_function_obs_1)
           de_orig = np.sum((all_new_function_obs_1 -
                             all_new_function_expected_1) ** 2.)
           de_corr = np.sum((all_new_function_obs_1 - y_c) ** 2.)
             
           new_kappa=de_orig/de_corr
             
           # Store kappa ratio per gmpe
           kappa_ratio.append(new_kappa/original_kappa)
    
        # Check kappa ratios
        if np.all(kappa_ratio) != 1.0:
            raise ValueError('kappa values do not match')
        else:
            pass

    def tearDown(self):
        """
        Remove outputs
        """
        shutil.rmtree(self.output_database)
        shutil.rmtree(self.output_directory)