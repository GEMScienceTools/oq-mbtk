"""
:module:`openquake.sub.test.misc.edge_test`
"""

import os
import glob

import numpy as np
import unittest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from openquake.hazardlib.geo.geodetic import distance
from openquake.hazardlib.geo.mesh import Mesh

from openquake.sub.misc.edge import (_read_edge, _resample_edge,
                                     create_from_profiles, create_faults,
                                     _rotate_vector, line_between_two_points,
                                     _get_mean_longitude)
from openquake.sub.misc.profile import _read_profile


BASE_DATA_PATH = os.path.dirname(__file__)


class CreateFaultTest(unittest.TestCase):

    def setUp(self):
        path = os.path.join(BASE_DATA_PATH, '..', 'data', 'misc', 'top_mesh')
        x = np.loadtxt(os.path.join(path, 'top_mesh.x'))
        y = np.loadtxt(os.path.join(path, 'top_mesh.y'))
        z = np.loadtxt(os.path.join(path, 'top_mesh.z'))
        self.mesh = np.stack((x, y, z), 2)

    def test_create_virtual_fault(self):
        """
        Create profiles for the virtual fault and check that all are defined
        """
        thickness = 50.
        angles = [30., 45., 90., 135]
        sampling = 5
        idx = 0
        for angl in angles:
            lines = create_faults(self.mesh, idx, thickness, angl, sampling)
            for l in lines[0]:
                pts = [[p.longitude, p.latitude, p.depth] for p in l.points]
                pts = np.array(pts)
                self.assertTrue(not np.any(np.isnan(pts)))

        if False:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            fig = plt.figure()

            ax.plot(self.mesh[idx, :, 0], self.mesh[idx, :, 1],
                    self.mesh[idx, :, 2]*0.1, '-', lw=2)

            for angl in angles:
                lines = create_faults(self.mesh, 0, thickness, angl, sampling)
                col = np.random.rand(3)
                for l in lines[0]:
                    pts = [[p.longitude, p.latitude, p.depth] for
                           p in l.points]
                    pts = np.array(pts)
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2]*0.1, '-',
                            color=col)
            plt.show()


class MeanLongitudeTest(unittest.TestCase):

    def test_values_across_idl(self):
        computed = _get_mean_longitude(np.array([178, -179]))
        expected = 179.5
        np.testing.assert_equal(computed, expected)

    def test_values_simple(self):
        computed = _get_mean_longitude(np.array([178, 179]))
        expected = 178.5
        np.testing.assert_equal(computed, expected)


class Line3d2PointsTest(unittest.TestCase):

    def test01(self):
        pnt1 = np.array([1., 2., 3.])
        pnt2 = np.array([4., 5., 6.])
        expected = np.array([0.58, 0.58, 0.58])
        computed = line_between_two_points(pnt1, pnt2)
        np.testing.assert_allclose(computed, expected, rtol=1)


class RotateVectorTest(unittest.TestCase):
    """
    The tests are performed against the results computed with
    this tool:
    http://www.nh.cas.cz/people/lazar/celler/online_tools.php
    """

    def test01(self):
        """
        Rotate the x-axis of 45° around the y-axis
        """
        v = np.array([1, 0, 0])
        k = np.array([0, 1, 0])
        angle = 45
        computed = _rotate_vector(v, k, angle)
        expected = np.array([0.707107, 0, -0.707107])
        np.testing.assert_allclose(computed, expected, rtol=1)

    def test02(self):
        """
        More general case of rotation
        """
        v = np.array([0.4, 0.6, 0.2])
        k = np.array([0.5, 0.1, -0.4])
        angle = 53.
        computed = _rotate_vector(v, k, angle)
        expected = np.array([0.646455, 0.057751, 0.372506])
        np.testing.assert_allclose(computed, expected, rtol=1)


class CreateFromProfilesTest(unittest.TestCase):

    # TODO:
    # - check duplicated points in an edge
    # - manage the case of discontinuos edges

    def setUp(self):
        #
        path = os.path.join(BASE_DATA_PATH, '../data/slab/cs02/*.csv')
        self.profiles = []
        for filename in sorted(glob.glob(path)):
            self.profiles.append(_read_profile(filename))
        #
        path = os.path.join(BASE_DATA_PATH, '../data/slab/cs03/*.csv')
        self.profiles1 = []
        for filename in sorted(glob.glob(path)):
            self.profiles1.append(_read_profile(filename))
        #
        path = os.path.join(BASE_DATA_PATH, '../data/slab/cs04/*.csv')
        self.profiles2 = []
        for filename in sorted(glob.glob(path)):
            self.profiles2.append(_read_profile(filename))
        #
        path = os.path.join(BASE_DATA_PATH, '../data/profiles01/cs*.txt')
        self.profiles3 = []
        for filename in sorted(glob.glob(path)):
            self.profiles3.append(_read_profile(filename))

    def test_create0(self):
        """
        Create edges from profiles 0
        """
        # sampling: profile, edge
        msh = create_from_profiles(self.profiles, 10, 5, False)

    def test_create1(self):
        """
        Create edges from profiles 1
        """
        # sampling: profile, edge
        msh = create_from_profiles(self.profiles1, 5, 5, False)

    def test_create2(self):
        """
        Create edges from profiles 2
        """
        # sampling: profile, edge
        msh = create_from_profiles(self.profiles2, 20, 25, False)

    def test_create3(self):
        """
        Create edges from profiles 3
        """
        # sampling: profile, edge
        msh = create_from_profiles(self.profiles2, 50, 50, False)

    def _test_create4(self):
        """
        Create edges from profiles 3
        """
        msh = create_from_profiles(self.profiles3, 5, 5, False)
        assert not np.any(np.isnan(msh))


class ResampleEdgeTest(unittest.TestCase):

    def setUp(self):
        filename = os.path.join(BASE_DATA_PATH,
                                '../data/slab/edge/edge_000.csv')
        self.edge = _read_edge(filename)

    def test_edge_resampling01(self):
        """
        Test edge resampling with a resampling distance of 25 km
        """
        #
        # resampled profile
        sampling_distance = 25.
        out_line, _, _ = _resample_edge(self.edge, sampling_distance, 5)
        #
        # lists with coordinates for the resampled profile
        lo = [pnt.longitude for pnt in out_line.points]
        la = [pnt.latitude for pnt in out_line.points]
        de = [pnt.depth for pnt in out_line.points]
        #
        # lenghts of resampled segments
        dsts = []
        for i in range(0, len(out_line)-1):
            dsts.append(distance(lo[i], la[i], de[i],
                                 lo[i+1], la[i+1], de[i+1]))
        #
        # testing
        expected = np.ones((len(out_line)-1))*sampling_distance
        np.testing.assert_allclose(dsts, expected, rtol=2, atol=0.)

    def test_edge_resampling02(self):
        """
        Test edge resampling with a resampling distance of 10 km
        """
        #
        # resampled profile
        sampling_distance = 10.
        out_line, _, _ = _resample_edge(self.edge, sampling_distance, 5)
        #
        # lists with coordinates for the resampled profile
        lo = [pnt.longitude for pnt in out_line.points]
        la = [pnt.latitude for pnt in out_line.points]
        de = [pnt.depth for pnt in out_line.points]
        #
        # lenghts of resampled segments
        dsts = []
        for i in range(0, len(out_line)-1):
            dsts.append(distance(lo[i], la[i], de[i],
                                 lo[i+1], la[i+1], de[i+1]))
        #
        # testing
        expected = np.ones((len(out_line)-1))*sampling_distance
        np.testing.assert_allclose(dsts, expected, rtol=2, atol=0.)


class ReadEdgeTest(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(BASE_DATA_PATH,
                                     '../data/slab/edge/edge_000.csv')

    def test_read_profile(self):
        """
        Test reading a edge file
        """
        computed = _read_edge(self.filename)
        lons = [pnt.longitude for pnt in computed.points]
        lats = [pnt.latitude for pnt in computed.points]
        deps = [pnt.depth for pnt in computed.points]

        lons_expected = [-8.294831883250239457e+01, -8.347113383616317606e+01,
                         -8.443028702759889370e+01, -8.505794151852860807e+01,
                         -8.584561547512082313e+01, -8.631551275344533281e+01,
                         -8.683238047673029314e+01, -8.776764521710948941e+01,
                         -8.890904386827106975e+01, -8.970302148270327791e+01,
                         -9.007321601251436505e+01, -9.098563317709692910e+01,
                         -9.202878921049629923e+01, -9.286755595092729720e+01,
                         -9.377193007159837634e+01, -9.467064876474159973e+01,
                         -9.573164826059495169e+01, -9.658845523814640899e+01,
                         -9.852944168622553889e+01, -1.002200364234107468e+02,
                         -1.010518388869808177e+02, -1.017966307049553194e+02,
                         -1.027087419628715566e+02, -1.034520970862245122e+02,
                         -1.043126646046702177e+02, -1.049145053002839632e+02,
                         -1.057032567476713325e+02]

        lats_expected = [7.655890711151086769e+00, 8.592405740147635029e+00,
                         8.926827693580914769e+00, 9.379254904438523610e+00,
                         9.800896181004983276e+00, 1.052077644719489413e+01,
                         1.126126700604738140e+01, 1.185098362267974181e+01,
                         1.216955938028376316e+01, 1.257674880493079073e+01,
                         1.288726010003954414e+01, 1.300458168978518714e+01,
                         1.364439121600205773e+01, 1.398627418090333485e+01,
                         1.434332714654129859e+01, 1.488407045045097910e+01,
                         1.540204147420979730e+01, 1.576928904676865528e+01,
                         1.607833500594980691e+01, 1.668378236227314559e+01,
                         1.707899734826530036e+01, 1.744602440690043821e+01,
                         1.791135119785566232e+01, 1.816301943627114923e+01,
                         1.846663314884608553e+01, 1.893173126671553774e+01,
                         1.966107823770858332e+01]

        deps_expected = [1.181428571428580199e+01, 1.288571428571435717e+01,
                         7.885714285714357175e+00, 5.385714285714357175e+00,
                         1.002857142857152439e+01, 1.288571428571435717e+01,
                         1.574285714285718996e+01, 2.038571428571435717e+01,
                         1.074285714285718996e+01, 8.600000000000079581e+00,
                         1.431428571428580199e+01, 1.217142857142863477e+01,
                         1.145714285714291236e+01, 7.528571428571524393e+00,
                         1.145714285714291236e+01, 7.528571428571524393e+00,
                         4.671428571428634768e+00, 1.752857142857152439e+01,
                         5.028571428571524393e+00, 6.457142857142912362e+00,
                         6.100000000000079581e+00, 7.528571428571524393e+00,
                         7.885714285714357175e+00, 6.457142857142912362e+00,
                         6.814285714285801987e+00, 8.957142857142912362e+00,
                         7.528571428571524393e+00]

        np.testing.assert_almost_equal(lons, lons_expected)
        np.testing.assert_almost_equal(lats, lats_expected)
        np.testing.assert_almost_equal(deps, deps_expected)
