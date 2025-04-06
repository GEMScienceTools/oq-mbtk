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
Test suite for the `residual_plotter` module responsible for plotting the
plot data defined in `residual_plots`
"""
import os
import shutil
import unittest
import pickle
from unittest.mock import patch, MagicMock

from openquake.smt.residuals.parsers.esm_url_flatfile_parser import ESMFlatfileParserURL
import openquake.smt.residuals.gmpe_residuals as res
from openquake.smt.residuals.sm_database_visualiser import DISTANCES
from openquake.smt.residuals.residual_plotter import (ResidualPlot,
                                                      LikelihoodPlot,
                                                      ResidualWithMagnitude,
                                                      ResidualWithDepth,
                                                      ResidualWithVs30,
                                                      ResidualWithDistance)


BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


class ResidualsTestCase(unittest.TestCase):
    """
    Core test case for the residuals objects
    """

    @classmethod
    def setUpClass(cls):
        """
        Setup constructs the database from the ESM test data
        """
        ifile = os.path.join(BASE_DATA_PATH, "residual_tests_data.csv")
        cls.out_location = os.path.join(BASE_DATA_PATH, "residual_tests")
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

    @patch('openquake.smt.residuals.residual_plotter.plt.subplot')
    @patch('openquake.smt.residuals.residual_plotter.plt')
    def tests_residual_plotter(self, mock_pyplot, mock_pyplot_subplot):
        """
        Tests basic execution of residual plot.
        Simply tests pyplot show is called by mocking its `show` method
        """
        # setup a mock which will handle all calls to matplotlib Axes calls
        # (e.g., bar, plot or semilogx) so we can test what has been called:
        mocked_axes_obj = MagicMock()
        mock_pyplot_subplot.side_effect = lambda *a, **v: mocked_axes_obj

        residuals = res.Residuals(self.gsims, self.imts)
        residuals.compute_residuals(self.database, component="Geometric")

        for gsim in self.gsims:
            for imt in self.imts:
                ResidualPlot(residuals, gsim, imt, bin_width=0.1)
                # assert we have not called pyplot show:
                self.assertTrue(mock_pyplot.show.call_count == 0)
                ResidualPlot(residuals, gsim, imt, bin_width=0.1,
                             show=False)
                # assert still not called pyplot show (call count still 1):
                self.assertTrue(mock_pyplot.show.call_count == 0)
                # reset mock:
                mock_pyplot.show.reset_mock()

                # assert we called the right matplotlib plotting functions:
                self.assertTrue(mocked_axes_obj.bar.called)
                self.assertTrue(mocked_axes_obj.plot.called)
                self.assertFalse(mocked_axes_obj.semilogx.called)
                # reset mock:
                mocked_axes_obj.reset_mock()

    @patch('openquake.smt.residuals.residual_plotter.plt.subplot')
    @patch('openquake.smt.residuals.residual_plotter.plt')
    def tests_likelihood_plotter(self, mock_pyplot, mock_pyplot_subplot):
        """
        Tests basic execution of Likelihood plotD.
        Simply tests pyplot show is called by mocking its `show` method
        """
        # setup a mock which will handle all calls to matplotlib Axes calls
        # (e.g., bar, plot or semilogx) so we can test what has been called:
        mocked_axes_obj = MagicMock()
        mock_pyplot_subplot.side_effect = lambda *a, **v: mocked_axes_obj

        residuals = res.Residuals(self.gsims, self.imts)
        residuals.compute_residuals(self.database, component="Geometric")

        for gsim in self.gsims:
            for imt in self.imts:
                LikelihoodPlot(residuals, gsim, imt, bin_width=0.1)
                # assert we have not called pyplot show:
                self.assertTrue(mock_pyplot.show.call_count == 0)
                LikelihoodPlot(residuals, gsim, imt, bin_width=0.1,
                               show=False)
                # assert still not called pyplot show (call count still 0):
                self.assertTrue(mock_pyplot.show.call_count == 0)
                # reset mock:
                mock_pyplot.show.reset_mock()

                # assert we called the right matplotlib plotting functions:
                self.assertTrue(mocked_axes_obj.bar.called)
                self.assertFalse(mocked_axes_obj.plot.called)
                self.assertFalse(mocked_axes_obj.semilogx.called)
                # reset mock:
                mocked_axes_obj.reset_mock()

    @patch('openquake.smt.residuals.residual_plotter.plt.subplot')
    @patch('openquake.smt.residuals.residual_plotter.plt')
    def tests_with_mag_vs30_depth_plotter(self, mock_pyplot,
                                          mock_pyplot_subplot):
        """
        Tests basic execution of residual with (magnitude, vs30, depth) plots.
        Simply tests pyplot show is called by mocking its `show` method
        """
        # setup a mock which will handle all calls to matplotlib Axes calls
        # (e.g., bar, plot or semilogx) so we can test what has been called:
        mocked_axes_obj = MagicMock()
        mock_pyplot_subplot.side_effect = lambda *a, **v: mocked_axes_obj

        residuals = res.Residuals(self.gsims, self.imts)
        residuals.compute_residuals(self.database, component="Geometric")

        for gsim in self.gsims:
            for imt in self.imts:
                for plotClass in [ResidualWithMagnitude,
                                  ResidualWithDepth,
                                  ResidualWithVs30]:
                    plotClass(residuals, gsim, imt, bin_width=0.1)
                    # assert we have not called pyplot show:
                    self.assertTrue(mock_pyplot.show.call_count == 0)
                    plotClass(residuals, gsim, imt, bin_width=0.1, show=False)
                    # assert still not called pyplot show (call count still 0):
                    self.assertTrue(mock_pyplot.show.call_count == 0)
                    # reset mock:
                    mock_pyplot.show.reset_mock()

                    # assert we called the right matplotlib plotting functions:
                    self.assertFalse(mocked_axes_obj.bar.called)
                    self.assertTrue(mocked_axes_obj.plot.called)
                    self.assertFalse(mocked_axes_obj.semilogx.called)

                    # check plot type:
                    plotClass(residuals, gsim, imt, plot_type='log',
                              bin_width=0.1, show=False)
                    self.assertTrue(mocked_axes_obj.semilogx.called)

                    # reset mock:
                    mocked_axes_obj.reset_mock()

    @patch('openquake.smt.residuals.residual_plotter.plt.subplot')
    @patch('openquake.smt.residuals.residual_plotter.plt')
    def tests_with_distance(self, mock_pyplot, mock_pyplot_subplot):
        """
        Tests basic execution of residual with distance plots.
        Simply tests pyplot show is called by mocking its `show` method
        """
        # setup a mock which will handle all calls to matplotlib Axes calls
        # (e.g., bar, plot or semilogx) so we can test what has been called:
        mocked_axes_obj = MagicMock()
        mock_pyplot_subplot.side_effect = lambda *a, **v: mocked_axes_obj

        residuals = res.Residuals(self.gsims, self.imts)
        residuals.compute_residuals(self.database, component="Geometric")

        for gsim in self.gsims:
            for imt in self.imts:
                for dist in DISTANCES.keys():

                    if dist == 'r_x':
                        # as for residual_plots_test, we should confirm
                        # with scientific expertise that this is the case:
                        with self.assertRaises(AttributeError):
                            ResidualWithDistance(residuals, gsim, imt,
                                                 distance_type=dist,
                                                 show=False)
                        continue

                    ResidualWithDistance(residuals, gsim, imt, bin_width=0.1)
                    # assert we have not called pyplot show:
                    self.assertTrue(mock_pyplot.show.call_count == 0)
                    ResidualWithDistance(residuals, gsim, imt, bin_width=0.1,
                                         show=False)
                    # assert still not called pyplot show (call count still 0):
                    self.assertTrue(mock_pyplot.show.call_count == 0)
                    # reset mock:
                    mock_pyplot.show.reset_mock()

                    # assert we called the right matplotlib plotting functions:
                    self.assertFalse(mocked_axes_obj.bar.called)
                    self.assertTrue(mocked_axes_obj.plot.called)
                    self.assertFalse(mocked_axes_obj.semilogx.called)

                    # check plot type:
                    ResidualWithDistance(residuals, gsim, imt,
                                         plot_type='', bin_width=0.1,
                                         show=False)
                    self.assertTrue(mocked_axes_obj.plot.called)

                    # reset mock:
                    mocked_axes_obj.reset_mock()

    @classmethod
    def tearDownClass(cls):
        """
        Deletes the database
        """
        shutil.rmtree(cls.out_location)