"""
:module:`openquake.sub.tests.slab.rupture_test_south_america_slab6`
"""

import pathlib
import unittest
import shutil
import tempfile

from subprocess import call

from openquake.sub.slab.rupture import calculate_ruptures
from openquake.sub.create_inslab_nrml import create
from openquake.sub.build_complex_surface import build_complex_surface

HERE = pathlib.Path(__file__).parent
PLOTTING = False


class RuptureCreationMarianaTest(unittest.TestCase):

    def test_create(self):
        """
        Test rupture calculation
        """
        ini_fname = HERE / 'data' / 'mariana' / 'slab.ini'

        # Create the tmp objects
        out_path = pathlib.Path(tempfile.mkdtemp())
        out_hdf5_fname = out_path / 'ruptures.hdf5'
        out_hdf5_smoothing_fname = out_path / 'smoothing.hdf5'

        # Compute ruptures
        kwargs = {'out_hdf5_fname': out_hdf5_fname,
                  'out_hdf5_smoothing_fname': out_hdf5_smoothing_fname}
        calculate_ruptures(ini_fname, **kwargs)
