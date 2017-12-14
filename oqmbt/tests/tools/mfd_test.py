
import numpy
import unittest

from oqmbt.mfd_tools import mfd_upsample, mfd_downsample
from oqmbt.mfd_tools import EEvenlyDiscretizedMFD

from openquake.hazardlib.mfd import EvenlyDiscretizedMFD


class TestStackMFDs(unittest.TestCase):

    def setUp(self):
        self.mfd1 = EEvenlyDiscretizedMFD(4.5, 0.1, [0.5, 0.4, 0.3, 0.2])
        self.mfd2 = EEvenlyDiscretizedMFD(4.4, 0.1, [0.5, 0.4, 0.3, 0.2])
        self.mfd3 = EvenlyDiscretizedMFD(4.4, 0.05, [0.5, 0.4, 0.3, 0.2])
        self.mfd4 = EvenlyDiscretizedMFD(4.4, 0.1, [0.5, 0.4, 0.3, 0.2,
                                                    0.1, 0.05, 0.025, 0.01])
        self.mfd5 = EvenlyDiscretizedMFD(4.6, 0.1, [0.5, 0.4, 0.3, 0.2,
                                                    0.1, 0.05, 0.025, 0.01])

        self.base = EEvenlyDiscretizedMFD(6.0, 0.1, [1e-20])
        self.huas082 = EvenlyDiscretizedMFD(4.7, 0.2, [0.000890073609248,
                    0.000561598480883, 0.000354344686162, 0.000223576382212,
                    0.000141067160409, 8.90073609248e-05, 5.61598480883e-05,
                    3.54344686162e-05, 2.23576382212e-05, 1.41067160409e-05,
                    8.90073609248e-06, 2.8088205502e-06, 3.54240181229e-07,
                    2.23510444056e-07])

        self.mfd6 = EvenlyDiscretizedMFD(4.7, 0.01, [0.5])
        self.mfd7 = EvenlyDiscretizedMFD(5.5, 0.01, [0.5])

    def test_stack_01(self):
        self.mfd1.stack(self.mfd1)
        res = numpy.array(self.mfd1.get_annual_occurrence_rates())
        expected = numpy.array([[4.5, 1.0],
                                [4.6, 0.8],
                                [4.7, 0.6],
                                [4.8, 0.4],
                                ])
        print 'res'
        print sum(res[:, 1])
        print res
        print 'expected'
        print sum(expected[:, 1])
        self.assertTrue(numpy.allclose(res, expected))

    def test_stack_02(self):
        self.mfd1.stack(self.mfd2)
        res = numpy.array(self.mfd1.get_annual_occurrence_rates())
        print res
        expected = numpy.array([[4.4, 0.5],
                                [4.5, 0.9],
                                [4.6, 0.7],
                                [4.7, 0.5],
                                [4.8, 0.2]])
        self.assertTrue(numpy.allclose(res, expected))

    def test_stack_03(self):
        self.mfd1.stack(self.mfd3)
        res = numpy.array(self.mfd1.get_annual_occurrence_rates())
        expected = numpy.array([[4.4, 0.7],
                                [4.5, 1.1],
                                [4.6, 0.5],
                                [4.7, 0.3],
                                [4.8, 0.2]])
        print res
        print sum(res[:, 1])
        print 'expected',
        print sum(expected[:, 1])
        self.assertTrue(numpy.allclose(res, expected))

    def test_stack_04(self):
        self.mfd1.stack(self.mfd4)
        res = numpy.array(self.mfd1.get_annual_occurrence_rates())
        print res
        expected = numpy.array([[4.4, 0.5],
                                [4.5, 0.9],
                                [4.6, 0.7],
                                [4.7, 0.5],
                                [4.8, 0.3],
                                [4.9, 0.05],
                                [5.0, 0.025],
                                [5.1, 0.01]
                                ])
        self.assertTrue(numpy.allclose(res, expected))

    def test_stack_05(self):
        self.mfd1.stack(self.mfd5)
        res = numpy.array(self.mfd1.get_annual_occurrence_rates())
        print res
        expected = numpy.array([[4.4, 0.5],
                                [4.5, 0.9],
                                [4.6, 0.7],
                                [4.7, 0.5],
                                [4.8, 0.3],
                                [4.9, 0.05],
                                [5.0, 0.025],
                                [5.1, 0.01]
                                ])
        #  self.assertTrue(numpy.allclose(res, expected))


    def test_stack_06(self):
        self.mfd1.stack(self.mfd6)
        res = numpy.array(self.mfd1.get_annual_occurrence_rates())
        print 'res:',
        print res
        expected = numpy.array([[4.5, 0.5],
                                [4.6, 0.4],
                                [4.7, 0.8],
                                [4.8, 0.2],
                                ])
        self.assertTrue(numpy.allclose(res, expected))


    def test_stack_07(self):
        self.mfd1.stack(self.mfd7)
        res = numpy.array(self.mfd1.get_annual_occurrence_rates())
        print 'res:',
        print res
        expected = numpy.array([[4.5, 0.5],
                                [4.6, 0.4],
                                [4.7, 0.3],
                                [4.8, 0.2],
                                [4.9, 0.0],
                                [5.0, 0.0],
                                [5.1, 0.0],
                                [5.2, 0.0],
                                [5.3, 0.0],
                                [5.4, 0.0],
                                [5.5, 0.5],
                                ])
        self.assertTrue(numpy.allclose(res, expected))


"""
    def test_stack_share_01(self):
        self.base.stack(self.huas082)
        print self.huas082.get_min_max_mag()
        res = numpy.array(self.base.get_annual_occurrence_rates())
        print self.base.get_min_max_mag()
        print res
        expected = numpy.array(self.huas082.get_annual_occurrence_rates())
        self.assertTrue(numpy.allclose(res, expected))
        self.mfd6 = EvenlyDiscretizedMFD(4.7, 0.01, [0.5])
"""


"""
class TestStackMFDs(unittest.TestCase):

    def setUp(self):
        self.mfd = EvenlyDiscretizedMFD(4.5, 0.1, [0.5, 0.4, 0.3, 0.2])
        self.mfd1 = EvenlyDiscretizedMFD(4.4, 0.1, [0.5, 0.4, 0.3, 0.2])
        self.mfd2 = EvenlyDiscretizedMFD(4.4, 0.05, [0.5, 0.4, 0.3, 0.2])

    def test_stack_01(self):
        out_mfd = mfd_stack(self.mfd, self.mfd)
        res = numpy.array(out_mfd.get_annual_occurrence_rates())
        print res
        expected = numpy.array([[4.5, 1.0],
                                [4.6, 0.8],
                                [4.7, 0.6],
                                [4.8, 0.4]])
        self.assertTrue(numpy.array_equal(res, expected))

    def test_stack_02(self):
        out_mfd = mfd_stack(self.mfd, self.mfd1)
        res = numpy.array(out_mfd.get_annual_occurrence_rates())
        print res
        expected = numpy.array([[4.4, 0.5],
                                [4.5, 0.9],
                                [4.6, 0.7],
                                [4.7, 0.5],
                                [4.8, 0.2]])
        self.assertTrue(numpy.allclose(res, expected))

    def test_stack_03(self):
        out_mfd = mfd_stack(self.mfd, self.mfd2)
        res = numpy.array(out_mfd.get_annual_occurrence_rates())
        print res
        expected = numpy.array([[4.4, 0.5],
                                [4.5, 0.9],
                                [4.6, 0.7],
                                [4.7, 0.5],
                                [4.8, 0.2]])
        #self.assertTrue(numpy.allclose(res, expected))

"""


class TestUpsampleMFD(unittest.TestCase):

    def setUp(self):
        self.mfd = EvenlyDiscretizedMFD(4.4, 0.05, [0.5, 0.4, 0.3, 0.2])

    def test_upsample_01(self):
        out_mfd = mfd_upsample(0.1, self.mfd)
        res = numpy.array(out_mfd.get_annual_occurrence_rates())
        expected = numpy.array([[4.4, 0.7],
                                [4.5, 0.6],
                                [4.6, 0.1]])
        self.assertTrue(numpy.allclose(res, expected))

    def test_upsample_02(self):
        out_mfd = mfd_upsample(0.15, self.mfd)
        res = numpy.array(out_mfd.get_annual_occurrence_rates())
        print res
        expected = numpy.array([[4.35, 0.5],
                                [4.5, 0.9]])
        self.assertTrue(numpy.allclose(res, expected))

    def test_upsample_03(self):
        out_mfd = mfd_upsample(0.12, self.mfd)
        res = numpy.array(out_mfd.get_annual_occurrence_rates())
        expected = numpy.array([[4.32, 0.05],
                                [4.44, 1.0],
                                [4.56, 0.35]])
        self.assertTrue(numpy.allclose(res, expected))


class TestDownsampleMFD(unittest.TestCase):

    def setUp(self):
        self.mfd = EvenlyDiscretizedMFD(4.4, 0.1, [0.5, 0.4, 0.3, 0.2])
        self.mfd1 = EvenlyDiscretizedMFD(4.4, 0.1, [0.5, 0.4])
        self.mfd2 = EvenlyDiscretizedMFD(4.05, 0.1, [0.5, 0.4])

        # This is the MFD for source HUAS082 in the SHARE (2013) model
        # with just area sources
        rates = [0.000890073609248, 0.000561598480883, 0.000354344686162,
                 0.000223576382212, 0.000141067160409, 8.90073609248e-05,
                 5.61598480883e-05, 3.54344686162e-05, 2.23576382212e-05,
                 1.41067160409e-05, 8.90073609248e-06, 2.8088205502e-06,
                 3.54240181229e-07, 2.23510444056e-07]
        self.mfd3 = EvenlyDiscretizedMFD(4.7, 0.2, rates)

    def test_downsample_01(self):
        out_mfd = mfd_downsample(0.05, self.mfd)
        res = numpy.array(out_mfd.get_annual_occurrence_rates())
        print res
        expected = numpy.array([[4.35, 0.125],
                                [4.40, 0.250],
                                [4.45, 0.225],
                                [4.50, 0.200],
                                [4.55, 0.175],
                                [4.60, 0.150],
                                [4.65, 0.125],
                                [4.70, 0.100],
                                [4.75, 0.050],
                                ])
        self.assertTrue(numpy.allclose(res, expected))

    def test_downsample_02(self):
        out_mfd = mfd_downsample(0.06, self.mfd1)
        res = numpy.array(out_mfd.get_annual_occurrence_rates())
        expected = numpy.array([[4.38, 0.30],
                                [4.44, 0.28],
                                [4.50, 0.24],
                                [4.56, 0.08],
                                ])
        self.assertTrue(numpy.allclose(res, expected))

    def test_downsample_03(self):
        out_mfd = mfd_downsample(0.1, self.mfd2)
        res = numpy.array(out_mfd.get_annual_occurrence_rates())
        print res
        expected = numpy.array([[4.0, 0.25],
                                [4.1, 0.45],
                                [4.2, 0.2],
                                ])
        self.assertTrue(numpy.allclose(res, expected))

    def test_downsample_04(self):
        out_mfd = mfd_downsample(0.1, self.mfd3)
        res = numpy.array(out_mfd.get_annual_occurrence_rates())
        print res
        expected = numpy.array([[4.0, 0.25],
                                [4.1, 0.45],
                                [4.2, 0.2],
                                ])
        # self.assertTrue(numpy.allclose(res, expected))
