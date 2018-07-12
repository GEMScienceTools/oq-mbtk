
import os
import shutil
import unittest

from openquake.sub.build_complex_surface import build_complex_surface
from openquake.sub.create_2pt5_model import read_profiles_csv


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class BuildComplexSurfaceTest(unittest.TestCase):

    def setUp(self):
        #
        # set the input folder
        self.in_path = os.path.join(DATA_PATH, 'cs_cam')
        #
        # set the output folder
        out_path = os.path.join(DATA_PATH, 'tmp')
        #
        # cleaning the temporary output folder
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path)
        #
        # maximum sampling distance
        self.out_path = out_path
        self.max_sampl_dist = 20

    def test_build_01(self):
        """
        """
        upper_depth = 0
        lower_depth = 1000
        from_id = '.*'
        to_id = '.*'
        #
        # read profiles
        sps, odmin, odmax = read_profiles_csv(self.in_path, upper_depth,
                                              lower_depth, from_id, to_id)
        #
        # build the complex surface
        build_complex_surface(self.in_path, self.max_sampl_dist, self.out_path,
                              upper_depth, lower_depth, from_id, to_id)
        #
        # read the output profiles
        sps, edmin, edmax = read_profiles_csv(self.out_path, upper_depth,
                                              lower_depth, from_id, to_id)
        #
        #
        self.assertEqual(odmin, edmin)
        assert lower_depth >= edmax
        assert upper_depth <= edmin
