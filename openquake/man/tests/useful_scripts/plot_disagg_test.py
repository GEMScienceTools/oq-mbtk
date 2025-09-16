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

import os
import shutil
import unittest

from openquake.man.utilities.plot_3d_disagg import disagg_MRE, disagg_MLL, disagg_TLL


base = os.path.dirname(__file__)


class TestPlotDisaggMRE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fname = os.path.join(base, "data_plot_disagg_test", "calc_767.hdf5")
        cls.out_mre = os.path.join(base, "..", "..", "useful_scripts", "disagg_Mag_Dist_Eps_calc_767")
        cls.out_mll = os.path.join(base, "..", "..", "useful_scripts", "disagg_Mag_Lon_Lat_calc_767")
        cls.out_tll = os.path.join(base, "..", "..", "useful_scripts", "disagg_TRT_Lon_Lat_calc_767")

    def test_plot_disagg_MRE(self):
        """
        Check execution of 3D mag-dist-eps plotting function
        """
        disagg_MRE(self.fname, "Mag_Dist_Eps", None, -30)

    def test_plot_disagg_MLL(self):
        """
        Check execution of 3D mag-lon-lat plotting function
        """
        disagg_MLL(self.fname, "Mag_Lon_Lat", None, -30)

    def test_plot_disagg_TLL(self):
        """
        Check execution of 3D trt-lon-lat plotting function
        """
        disagg_TLL(self.fname, "TRT_Lon_Lat", None, -30)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out_mre)
        shutil.rmtree(cls.out_mll)
        shutil.rmtree(cls.out_tll)

