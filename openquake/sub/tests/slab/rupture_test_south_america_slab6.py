"""
:module:`openquake.sub.tests.slab.rupture_test_south_america_slab6`
"""

import os
import unittest
import shutil

from openquake.sub.slab.rupture import calculate_ruptures
from openquake.sub.create_inslab_nrml import create
from openquake.sub.build_complex_surface import build_complex_surface

BASE_DATA_PATH = os.path.dirname(__file__)


class RuptureCreationSouthAmericaTest(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        """
        relpath = '../data/ini/test_south_america_slab_6.ini'
        self.ini_fname = os.path.join(BASE_DATA_PATH, relpath)
        #
        # Prepare the input folder and the output folder
        tmps = '../data/south_america_segment6_slab/'
        in_path = os.path.join(BASE_DATA_PATH, tmps)
        self.out_path = os.path.join(BASE_DATA_PATH, '../data/tmp/')
        #
        # Cleaning the tmp directory
        if os.path.exists(self.out_path):
            shutil.rmtree(self.out_path)
        #
        # create the complex surface. We use the profiles used for
        # the subduction in CCARA
        max_sampl_dist = 10.
        build_complex_surface(in_path, max_sampl_dist, self.out_path,
                              upper_depth=50, lower_depth=200)

    def test_create(self):
        """
        Test rupture calculation
        """
        reff = os.path.join(BASE_DATA_PATH, '../data/ini/')
        calculate_ruptures(self.ini_fname, False, reff)

        label = 'test'
        tmps = '../data/tmp/ruptures.hdf5'
        rupture_hdf5_fname = os.path.abspath(os.path.join(BASE_DATA_PATH,
                                                          tmps))
        output_folder = os.path.join(BASE_DATA_PATH, self.out_path)
        investigation_t = '1.'

        # This is basically test 1
        reff = os.path.join(BASE_DATA_PATH, '../data/ini/')
        calculate_ruptures(self.ini_fname, False, reff)

        create(label, rupture_hdf5_fname, output_folder, investigation_t)
