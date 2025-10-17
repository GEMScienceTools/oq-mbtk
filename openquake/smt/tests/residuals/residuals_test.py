# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation and G. Weatherill
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
import sys
import ast
import pprint
import shutil
import tempfile
import unittest
import pickle
import numpy as np

import openquake.smt.residuals.gmpe_residuals as res
import openquake.smt.residuals.residual_plotter as rspl
from openquake.smt.residuals.parsers.esm_url_flatfile_parser import (
    ESMFlatfileParserURL)


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

TMP_TAB = os.path.join(tempfile.mkdtemp(), 'temp_table.csv')
TMP_FIG = os.path.join(tempfile.mkdtemp(), 'temp_figure.png')

aac = np.testing.assert_allclose


def fix(number):
    if np.isnan(number):
        return None
    else:
        return float(number)


def compare_residuals(kind, observed, expected):
    """
    Compare lists of triple dictionaries gsim -> imt -> key -> values
    """
    tmpdir = tempfile.mkdtemp()
    Result(tmpdir).save(observed)
    try:
        for idx, (obs, exps) in enumerate(zip(observed, expected)):
            for gsim, ddic in exps.items():
                for imt, dic in ddic.items():
                    for key, exp in dic.items():
                        got = obs[gsim][imt][key]
                        if not hasattr(exp, '__len__'):
                            exp = [exp]
                            got = [got]
                        for i, x in enumerate(exp):
                            if x is not None:
                                aac(got[i], exp[i], atol=1e-8,
                                    err_msg=f'in {gsim}-{idx}-{imt}-{key}-{i}')
    except Exception:
        print(f'Hint: meld {kind} {tmpdir}', file=sys.stderr)
        raise
    else:
        shutil.rmtree(tmpdir)


class Result:
    """
    Logic to read and save the residuals as .py data files
    """
    def __init__(self, dname):
        self.dname = dname

    def save(self, dddics):
        if not os.path.exists(self.dname):
            os.mkdir(self.dname)
        for i, dddic in enumerate(dddics):
            for gsim, ddic in dddic.items():
                with open(self.dname + f'/{gsim}-{i}.py', 'w') as f:
                    for k1, dic in ddic.items():
                        for k2, vals in dic.items():
                            if isinstance(vals, np.ndarray):
                                dic[k2] = [fix(x) for x in vals]
                            else:
                                dic[k2] = fix(vals)
                    pprint.pprint(ddic, f)
                    print(f'Saved {f.name}', file=sys.stderr)

    def read(self, gsim, idx=0):
        for fname in os.listdir(self.dname):
            if fname.startswith(gsim) and fname.endswith(f'-{idx}.py'):
                with open(os.path.join(self.dname, fname)) as f:
                    js = f.read()
                    return ast.literal_eval(js)


GSIMS = ['KothaEtAl2020', 'LanzanoEtAl2019_RJB_OMO']
CWD = os.path.dirname(__file__)
res1 = Result(os.path.join(CWD, 'exp_regular'))
exp = {gsim: res1.read(gsim) for gsim in GSIMS}
res2 = Result(os.path.join(CWD, 'exp_stations'))
exp_stations = [{gsim: res2.read(gsim, idx) for gsim in GSIMS}
                for idx in range(8)]


class ResidualsTestCase(unittest.TestCase):
    """
    Core test case for the residuals objects
    """

    @classmethod
    def setUpClass(cls):
        """
        Setup constructs the database from the ESM test data
        """
        # Make the database
        ifile = os.path.join(BASE_DATA_PATH, "residual_tests_data.csv")
        cls.out_location = os.path.join(BASE_DATA_PATH, "residual_tests")
        if os.path.exists(cls.out_location):
            shutil.rmtree(cls.out_location)
        parser = ESMFlatfileParserURL.autobuild(
            "000", "ESM_test_subset", cls.out_location, ifile)
        del parser
        cls.database_file = os.path.join(cls.out_location,
                                         "metadatafile.pkl")
        with open(cls.database_file, "rb") as f:
            cls.database = pickle.load(f)

        # Add the GMPE list and IMTs
        cls.imts = ["PGA", "SA(1.0)"]

        # Compute residuals here to avoid repeating in each test
        cls.residuals = res.Residuals(GSIMS, cls.imts)
        cls.residuals.compute_residuals(cls.database, component="Geometric")
        cls.residuals.get_residual_statistics()

        # Add other params to class
        cls.toml = os.path.join(BASE_DATA_PATH, 'residuals_from_toml_test.toml')
        cls.exp = exp
        cls.st_rec_min = 3
        cls.exp_stations = exp_stations

    def test_residual_values(self):
        """
        Check correctness of values for computed residuals
        """
        compare_residuals('exp', [self.residuals.residuals], [self.exp])

    def test_residuals_execution_from_toml(self):
        """
        Tests basic execution of residuals when specifying gmpe and imts to get
        residuals for from a toml file - not correctness of values
        """
        residuals = res.Residuals.from_toml(self.toml)
        residuals.compute_residuals(self.database, component="Geometric")
        residuals.get_residual_statistics()

    def test_export_execution(self):
        """
        Tests execution of the residuals exporting function
        """
        out_loc = os.path.join(self.out_location, "residuals.txt")
        self.residuals.export_residuals(out_loc)

    def test_likelihood_execution(self):
        """
        Tests basic execution of likelihood score (Scherbaum et al.
        2004) computation- not correctness of values
        """
        self.residuals.get_likelihood_values()

    def test_llh_execution(self):
        """
        Tests basic execution of loglikelihood score (Scherbaum et al.
        2009) computation- not correctness of values
        """
        self.residuals.get_llh_values()

    def test_edr_execution(self):
        """
        Tests basic execution of EDR score (Scherbaum et al.
        2004) computation- not correctness of values
        """
        self.residuals.get_edr_values()
          
    def test_stochastic_area_execution(self):
        """
        Tests basic execution of stochastic area metric scores (Sunny
        et al. 2021) computation - not correctness of values
        """
        self.residuals.get_sto_wrt_imt()

    def test_plot_execution(self):
        """
        Tests execution of gmpe ranking metric plotting functions and
        the means and stddevs plotting function
        """
        # First compute the metrics
        self.residuals.get_llh_values()
        self.residuals.get_edr_wrt_imt()
        self.residuals.get_sto_wrt_imt()

        # Make the plots
        rspl.plot_residual_means_and_stds_with_period(self.residuals, TMP_FIG)
        rspl.plot_edr_with_period(self.residuals, TMP_FIG)
        rspl.plot_llh_with_period(self.residuals, TMP_FIG)

    def test_table_execution(self):
        """
        Tests execution of table exporting functions
        """
        # First compute the metrics
        self.residuals.get_llh_values()
        self.residuals.get_edr_wrt_imt()
        self.residuals.get_sto_wrt_imt()
        
        # Tables of values
        rspl.residual_means_and_stds_table(self.residuals, TMP_TAB)
        rspl.llh_table(self.residuals, TMP_TAB)
        rspl.edr_table(self.residuals, TMP_TAB)
        rspl.sto_table(self.residuals, TMP_TAB)
        
        # Tables of weights
        rspl.llh_weights_table(self.residuals, TMP_TAB)
        rspl.edr_weights_table(self.residuals, TMP_TAB)
        rspl.sto_weights_table(self.residuals, TMP_TAB)
        
    def test_single_station_execution_and_values(self):
        """
        Test execution of single station residual analysis functions and
        correctness of values. Execution of plots is also tested here.
        """
        # Get sites with at least 3 record each
        top_sites = sorted(
            self.database.rank_sites_by_record_count(self.st_rec_min))
            
        # Create SingleStationAnalysis object
        ssa1 = res.SingleStationAnalysis(top_sites, GSIMS, self.imts)
        
        # Compute total, inter-event and intra-event residuals for each site
        ssa1.get_site_residuals(self.database)

        # Get station residual statistics per GMPE and per imt
        ssa_csv_output = os.path.join(self.out_location, 'ssa_test.csv')
        ssa1.station_residual_statistics(ssa_csv_output)
        
        # Check exp vs obs delta_s2ss, delta_woes, phi_ss per station
        compare_residuals(
            'stations',
            [stat.site_analysis for stat in ssa1.site_residuals],
            exp_stations)

        # Check num. sites, GMPEs and intensity measures + csv outputted
        self.assertTrue(len(ssa1.site_ids) == len(top_sites))
        self.assertTrue(len(ssa1.gmpe_list) == len(GSIMS))
        self.assertTrue(len(ssa1.imts) == len(self.imts))
        self.assertTrue(ssa_csv_output)
        
        # Check plots executed for each GMPE and intensity measure
        for gmpe in GSIMS:
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

    def test_single_station_execution_from_toml(self):
        """
        Test execution of single station residual analysis using GMPEs and
        imts specified within a toml file. Correctness of values is not tested.
        """
        # Get sites with at least 3 record each
        top_sites = self.database.rank_sites_by_record_count(self.st_rec_min)
        
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
