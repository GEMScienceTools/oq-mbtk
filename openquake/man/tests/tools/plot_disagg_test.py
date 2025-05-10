import os
import shutil
import unittest

from openquake.man.tools.plot_3d_disagg import disagg_MRE


base = os.path.dirname(__file__)


class TestPlotDisaggMRE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.in_mre = os.path.join(base, "data", "calc_763.hdf5")
        cls.out_mre = os.path.join(base, "..", "..", "tools", "disagg_Mag_Dist_Eps_calc_763")

    def test_plot_disagg_MRE(self):
        """
        Check execution of 3D mag-dist-eps plotting function
        """
        disagg_MRE(self.in_mre, "Mag_Dist_Eps", None, 45)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out_mre)
