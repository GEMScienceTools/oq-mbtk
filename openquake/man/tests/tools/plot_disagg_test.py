import os
import shutil
import unittest
import subprocess

from openquake.man.tools.plot_3d_disagg import disagg_MRE


base = os.path.abspath("")

out_mre = os.path.join(base, "..", "..", "tools", "disagg_Mag_Dist_Eps_calc_763")


class TestPlotDisaggMRE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.out_mre = out_mre

    def test_plot_disagg_MRE(self):
        """
        Check execution of 3D mag-dist-eps plotting function
        """
        disagg_MRE(763, "Mag_Dist_Eps", None, 45)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out_mre)

