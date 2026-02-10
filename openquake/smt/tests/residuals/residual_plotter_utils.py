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
Test suite for the `residual_plotter_utils` module responsible for
calculating the data used for plotting (see `residual_plotter`).
"""
import os
import shutil
import unittest
import pickle
import numpy as np
from scipy.stats import linregress

from openquake.smt.residuals.parsers.esm_url_flatfile_parser import ESMFlatfileParserURL
import openquake.smt.residuals.gmpe_residuals as res
from openquake.smt.residuals.sm_database_visualiser import DISTANCES
from openquake.smt.residuals.residual_plotter_utils import (residuals_density_distribution,
                                                    likelihood,
                                                    residuals_with_depth,
                                                    residuals_with_magnitude,
                                                    residuals_with_vs30,
                                                    residuals_with_distance,
                                                    _nanlinregress)


BASE = os.path.join(os.path.dirname(__file__), "data")


class ResidualsTestCase(unittest.TestCase):
    """
    Core test case for the residuals objects.
    """
    @classmethod
    def setUpClass(cls):
        """
        Setup constructs the database from the ESM test data.
        """
        ifile = os.path.join(BASE, "residual_tests_data.csv")
        cls.out_location = os.path.join(BASE, "residual_tests")
        if os.path.exists(cls.out_location):
            shutil.rmtree(cls.out_location)
        parser = ESMFlatfileParserURL.autobuild(
            "000", "ESM ALL", cls.out_location, ifile)
        del parser
        cls.database_file = os.path.join(cls.out_location,
                                         "metadatafile.pkl")
        cls.database = None
        with open(cls.database_file, "rb") as f:
            cls.database = pickle.load(f)
        cls.gsims = ["AkkarEtAlRjb2014",  "ChiouYoungs2014"]
        cls.imts = ["PGA", "SA(1.0)"]

    def _plot_data_check(self, plot_data,
                         expected_xlabel, expected_ylabel,
                         additional_keys=None):
        allkeys = ['x', 'y', 'xlabel', 'ylabel'] + \
            ([] if not additional_keys else list(additional_keys))

        for res_type in plot_data:
            res_data = plot_data[res_type]
            # assert we have the specified keys:
            self.assertTrue(len(res_data['x']) == len(res_data['y']))
            self.assertTrue(res_data['xlabel'] == expected_xlabel)
            self.assertTrue(res_data['ylabel'] == expected_ylabel)

    def _hist_data_check(self, residuals, gsim, imt, plot_data, bin_width):
        for res_type, res_data in plot_data.items():
            pts = residuals.residuals[gsim][imt][res_type]
            # FIXME: test below should be improved, it does not prevent
            # "false negatives":
            self.assertTrue(len(res_data['x']) != len(pts))

    def _scatter_data_check(self, residuals, gsim, imt, plot_data):
        for res_type, res_data in plot_data.items():
            assert len(residuals.residuals[gsim][imt][res_type]) == len(res_data['x'])

    def test_residual_density_distribution(self):
        """
        Tests basic execution of residual plot data - does not
        test correctness of values.
        """
        residuals = res.Residuals(self.gsims, self.imts)
        residuals.compute_residuals(self.database, component="Geometric")
        additional_keys = ['mean', 'stddev']
        bin_w1, bin_w2 = 0.1, 0.2
        
        for gsim in self.gsims:
            for imt in self.imts:
                data1 = residuals_density_distribution(
                    residuals, gsim, imt, bin_width=bin_w1)
                self._plot_data_check(
                    data1, "Z (%s)" % imt, "Frequency", additional_keys)
                data2 = residuals_density_distribution(
                    residuals, gsim, imt, bin_width=bin_w2)
                self._plot_data_check(
                    data2, "Z (%s)" % imt, "Frequency", additional_keys)

                # assert histogram data is ok:
                self._hist_data_check(residuals, gsim, imt, data1, bin_w1)
                self._hist_data_check(residuals, gsim, imt, data2, bin_w2)

                # assert bin width did its job:
                for res_type in data1:
                    self.assertTrue(len(data1[res_type]['x']) >
                                    len(data2[res_type]['x']))

    def test_likelihood_density_distribution(self):
        """
        Tests basic execution of Likelihood plot data - does
        not test correctness of values.
        """
        residuals = res.Residuals(self.gsims, self.imts)
        residuals.compute_residuals(self.database, component="Geometric")
        additional_keys = ['median']
        bin_w1, bin_w2 = 0.1, 0.2

        for gsim in self.gsims:
            for imt in self.imts:
                data1 = likelihood(residuals, gsim, imt, bin_width=bin_w1)
                self._plot_data_check(
                    data1, "LH (%s)" % imt, "Frequency", additional_keys)
                data2 = likelihood(
                    residuals, gsim, imt,  bin_width=bin_w2)
                self._plot_data_check(
                    data2, "LH (%s)" % imt, "Frequency", additional_keys)

                # assert histogram data is ok:
                self._hist_data_check(residuals, gsim, imt, data1, bin_w1)
                self._hist_data_check(residuals, gsim, imt, data2, bin_w2)

                # assert bin width did its job:
                for res_type in data1:
                    self.assertTrue(len(data1[res_type]['x']) >
                                    len(data2[res_type]['x']))

    def test_residuals_vs_mag_depth_vs30(self):
        """
        Tests basic execution of Resiuals vs (magnitude, depth,
        vs30) plot data - does not test correctness of values.
        """
        residuals = res.Residuals(self.gsims, self.imts)
        residuals.compute_residuals(self.database, component="Geometric")
        residuals.get_likelihood_values()
        additional_keys = ['slope', 'intercept', 'pvalue']
        values = [
                  (residuals_with_depth, "Hypocentral Depth (km)"),
                  (residuals_with_magnitude, "Magnitude (Mw)"),
                  (residuals_with_vs30, "Vs30 (m/s)")
                  ]

        for gsim in self.gsims:
            for imt in self.imts:
                for func, expected_xlabel in values:

                    data1 = func(residuals, gsim, imt)
                    self._plot_data_check(
                        data1, expected_xlabel, "Z (%s)" % imt, additional_keys)

                    # assert histogram data is ok:
                    self._scatter_data_check(residuals, gsim, imt, data1)

    def test_residuals_vs_distance(self):
        """
        Tests basic execution of Resiuals vs distances plot
        data - does not test correctness of values.
        """
        residuals = res.Residuals(self.gsims, self.imts)
        residuals.compute_residuals(self.database, component="Geometric")
        residuals.get_likelihood_values()
        additional_keys = ['slope', 'intercept', 'pvalue']

        for gsim in self.gsims:
            for imt in self.imts:
                for dist in DISTANCES.keys():
                    data1 = residuals_with_distance(residuals, gsim, imt, dist)
                    self._plot_data_check(
                        data1, "%s (km)" % dist, "Z (%s)" % imt, additional_keys)

                    # Assert histogram data is ok
                    self._scatter_data_check(residuals, gsim, imt, data1)

    def test_nanlinregress(self):
        self._assert_linreg([1, 2], [3.5, -4], [1, 2], [3.5, -4])
        self._assert_linreg([1, np.nan], [3.5, -4], [1], [3.5])
        self._assert_linreg([1, 2], [np.nan, -4], [2], [4])
        self._assert_linreg([1, np.nan], [np.nan, -4], [np.nan], [np.nan])
        # a less edgy test case:
        self._assert_linreg([1, np.nan, 4.5, 6], [np.nan, -4, 11, 0.005], [4.5, 6], [11, 0.005])

    def _assert_linreg(self, nanx, nany, x, y):
        # nanx, nany: values for _nanlinreg.
        # x, y: values for scipy linreg.
        # Asserts the results are the same
        l_1 = linregress(np.asarray(x), np.asarray(y))
        l_2 = _nanlinregress(np.asarray(nanx), np.asarray(nany))

        if np.isnan(l_1.slope):
            self.assertTrue(np.isnan(l_2.slope))
        else:
            self.assertEqual(l_1.slope, l_2.slope)

        if np.isnan(l_1.intercept):
            self.assertTrue(np.isnan(l_2.intercept))
        else:
            self.assertEqual(l_1.intercept, l_2.intercept)

    @classmethod
    def tearDownClass(cls):
        """
        Deletes the database.
        """
        shutil.rmtree(cls.out_location)