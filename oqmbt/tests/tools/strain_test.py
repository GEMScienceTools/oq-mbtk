
import numpy as np
import unittest

BASE_DATA_PATH = os.path.dirname(__file__)

class TestComputeStrain(unittest.TestCase):

    def setUp(self):
        strain_file = os.path.join(BASE_DATA_PATH, '../data/tools/strain.hdf5')
        f = h5py.File(strain_date_model_hdf5_file, 'r')
        xxx = f['gsrm'].value
        f.close()

    def test_01(self):
        """
        """
        pass
