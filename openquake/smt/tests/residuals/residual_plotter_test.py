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
Test suite for the `residual_plotter` module responsible for
plotting the plot data defined in `residual_plotter_utils`.
"""
import os
import shutil
import tempfile
import unittest
import pickle

from openquake.smt.residuals.parsers.esm_url_flatfile_parser import ESMFlatfileParserURL
import openquake.smt.residuals.gmpe_residuals as res
from openquake.smt.residuals.sm_database_visualiser import DISTANCES
from openquake.smt.residuals.residual_plotter import (ResidualPlot,
                                                      ResidualWithMagnitude,
                                                      ResidualWithDepth,
                                                      ResidualWithVs30,
                                                      ResidualWithDistance)


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
        cls.metadata = os.path.join(BASE, "residual_tests")
        if os.path.exists(cls.metadata):
            shutil.rmtree(cls.metadata)
        parser = ESMFlatfileParserURL.autobuild("000", "ESM ALL", cls.metadata, ifile)
        del parser
        cls.database_file = os.path.join(cls.metadata, "metadatafile.pkl")
        with open(cls.database_file, "rb") as f:
            cls.database = pickle.load(f)
        cls.gsims = ["AkkarEtAlRjb2014",  "ChiouYoungs2014"]
        cls.imts = ["PGA", "SA(1.0)"]
        cls.residuals = res.Residuals(cls.gsims, cls.imts)
        cls.residuals.compute_residuals(cls.database, component="Geometric")
        cls.fname = tempfile.TemporaryFile()

    def tests_residual_plotter(self):
        """
        Tests basic execution of residual plot.
        """
        for gsim in self.gsims:
            for imt in self.imts:
                ResidualPlot(self.residuals, gsim, imt, self.fname, bin_width=0.1)

    def tests_with_mag_vs30_depth_plotter(self):
        """
        Tests basic execution of residual with (magnitude, vs30, depth) plots.
        """
        for gsim in self.gsims:
            for imt in self.imts:
                ResidualWithMagnitude(self.residuals, gsim, imt, self.fname, bin_width=0.1)
                ResidualWithDepth(self.residuals, gsim, imt, self.fname, bin_width=0.1)
                ResidualWithVs30(self.residuals, gsim, imt, self.fname, bin_width=0.1)
                  
    def tests_with_distance(self):
        """
        Tests basic execution of residual with distance plots.
        """
        for gsim in self.gsims:
            for imt in self.imts:
                for dist in DISTANCES.keys():

                    if dist == 'r_x':
                        # Should raise an error for r_x
                        with self.assertRaises(AttributeError):
                            ResidualWithDistance(self.residuals,
                                                 gsim,
                                                 imt,
                                                 self.fname,
                                                 distance_type=dist)
                        continue

                    ResidualWithDistance(self.residuals, gsim, imt, self.fname, bin_width=0.1)

    @classmethod
    def tearDownClass(cls):
        """
        Deletes the database.
        """
        shutil.rmtree(cls.metadata)