import os
import shutil
import unittest

from openquake.man.tools.plot_3d_disagg import disagg_MRE, disagg_MLL, disagg_TLL


base = os.path.dirname(__file__)


class TestPlotDisaggMRE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fname = os.path.join(base, "data", "calc_767.hdf5")
        cls.out = os.path.join(base, "..", "..", "tools", "disagg_Mag_Dist_Eps_calc_767")

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
        shutil.rmtree(cls.out)
