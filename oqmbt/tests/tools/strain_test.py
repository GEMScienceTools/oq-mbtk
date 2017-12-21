
import numpy as np
import unittest

BASE_DATA_PATH = os.path.dirname(__file__)

class TestComputeStrain(unittest.TestCase):

    def setUp(self):
        strain_file = os.path.join(BASE_DATA_PATH, '../data/tools/strain.hdf5')

    def test_get_cumulative(self):
        """
        Test the values of magnitude computed and the complementary 
        cumulative rates
        """
        mac, occ = get_cumulative(self.mfd)
        expected = np.array([6.5, 6.6, 6.7, 6.8, 6.9])
        np.testing.assert_array_almost_equal(np.array(mac), expected)
        incremental = np.array([6.503912286587971e-05, 5.166241165406994e-05,
                                4.103691225077654e-05, 3.259677806669462e-05,
                                2.589254117941671e-05])
        tocc = sum(incremental)
        self.assertAlmostEqual(tocc, occ[0])
        self.assertAlmostEqual(incremental[-1], occ[-1])
        self.assertAlmostEqual(sum(incremental[-2:]), occ[3])
