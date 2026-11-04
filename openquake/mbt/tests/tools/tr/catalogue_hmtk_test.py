import unittest
import numpy as np

from openquake.mbt.tools.tr.catalogue_hmtk import get_distances_from_surface


class _FakeCatalogue:
    def __init__(self, nel):
        self.data = {'longitude': np.linspace(10., 11., nel),
                     'latitude': np.linspace(40., 41., nel),
                     'depth': np.arange(nel, dtype=float)}


class _FakeSurface:
    """ Returns the depth of each point as its distance """
    def get_min_distance(self, mesh):
        return np.array(mesh.depths)


class GetDistancesFromSurfaceTestCase(unittest.TestCase):
    """
    Tests that distances are computed for every earthquake, including the
    last one when the catalogue is split into blocks
    """

    def test_block_sizes(self):
        for nel in [10, 4000, 4001, 8000, 8001, 9500]:
            cat = _FakeCatalogue(nel)
            dsts = get_distances_from_surface(cat, _FakeSurface())
            np.testing.assert_array_equal(dsts, cat.data['depth'],
                                          err_msg=f'nel = {nel}')
