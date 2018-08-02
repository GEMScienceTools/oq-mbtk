"""
"""

import os
import unittest
import shutil

from openquake.sub.slab.rupture import calculate_ruptures
from openquake.sub.create_inslab_nrml import create
from openquake.sub.build_complex_surface import build_complex_surface

BASE_DATA_PATH = os.path.dirname(__file__)


class RuptureCreationCAMTest(unittest.TestCase):
    """
    This set of tests check the construction of ruptures for inslab sources
    using the Central America subduction zone.
    """

    def setUp(self):
        relpath = '../data/ini/test.ini'
        self.ini_fname = os.path.join(BASE_DATA_PATH, relpath)
        #
        # prepare the input folder and the output folder
        in_path = os.path.join(BASE_DATA_PATH, '../data/sp_cam/')
        out_path = os.path.join(BASE_DATA_PATH, '../data/tmp/')
        #
        # cleaning the tmp directory
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        #
        # create the complex surface. We use the profiles used for
        # the subduction in CCARA
        max_sampl_dist = 10.
        build_complex_surface(in_path, max_sampl_dist, out_path,
                              upper_depth=50, lower_depth=200)

    def step01(self):
        reff = os.path.abspath(os.path.join(BASE_DATA_PATH, '../data/ini/'))
        calculate_ruptures(self.ini_fname, False, reff)
        #
        # check the existence of the rupture file
        tmps = '../data/tmp/ruptures.hdf5'
        rupture_hdf5_fname = os.path.abspath(os.path.join(BASE_DATA_PATH,
                                                          tmps))
        self.assertTrue(os.path.exists(rupture_hdf5_fname))

    def step02(self):
        label = 'test'
        tmps = '../data/tmp/ruptures.hdf5'
        rupture_hdf5_fname = os.path.abspath(os.path.join(BASE_DATA_PATH,
                                                          tmps))
        output_folder = os.path.join(BASE_DATA_PATH, '../tmp/')
        investigation_t = '1.'
        create(label, rupture_hdf5_fname, output_folder, investigation_t)

    def _steps(self):
        for name in sorted(dir(self)):
            if name.startswith("step"):
                yield name, getattr(self, name)

    def test_steps(self):
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
