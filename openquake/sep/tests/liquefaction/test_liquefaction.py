import os
import unittest

import numpy as np

from openquake.sep.liquefaction import (
    zhu_magnitude_correction_factor,
    zhu_liquefaction_probability_general,
    hazus_magnitude_correction_factor,
    hazus_groundwater_correction_factor,
    hazus_conditional_liquefaction_probability,
    hazus_liquefaction_probability,
)


class test_zhu_functions(unittest.TestCase):
    def test_zhu_magnitude_correction_factor(self):
        mags = np.array([6.0, 7.0, 8.0])
        test_res = np.array([0.5650244, 0.83839945, 1.18007706])
        np.testing.assert_array_almost_equal(
            zhu_magnitude_correction_factor(mags), test_res
        )

    def test_zhu_liquefaction_probability_general(self):
        pass


class test_hazus_liquefaction_functions(unittest.TestCase):
    def test_hazus_magnitude_correction_factor(self):
        # magnitudes selected to roughly replicate Fig. 4.7 in the Hazus manual
        mags = np.array([5.1, 6.1, 6.8, 7.6, 8.4])
        Km = hazus_magnitude_correction_factor(mags)
        test_res = np.array(
            [1.5344407, 1.2845917, 1.1357584, 1.0000432, 0.9089488]
        )
        np.testing.assert_array_almost_equal(Km, test_res)

    def test_hazus_gw_correction_factor_ft(self):
        # replicates Fig. 4.8 in the Hazus manual
        depth_ft = np.arange(4, 36, 4)
        Kw = hazus_groundwater_correction_factor(depth_ft)
        test_res = np.array([1.018, 1.25266667, 1.48733333, 1.722])
        np.testing.assert_array_almost_equal(Kw, test_res)

    def test_hazus_conditional_liquefaction_probability_vl(self):
        # replicates Fig. 4.6 in the Hazus manual
        pga_vl = np.linspace(0.2, 0.6, num=10)
        cond_vl = hazus_conditional_liquefaction_probability(pga_vl, "vl")
        test_res = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.01473684,
                0.10231579,
                0.18989474,
                0.27747368,
                0.36505263,
                0.45263158,
                0.54021053,
                0.62778947,
                0.71536842,
                0.80294737,
                0.89052632,
                0.97810526,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        )

        np.testing.assert_array_almost_equal(cond_vl, test_res)

    def test_hazus_conditional_liquefaction_probability_l(self):
        # Replicates Fig. 4.6 in the Hazus manual
        # However values do not match figure exactly, though
        # the formula and coefficients are double-checked...
        pga_l = np.linspace(0.2, 0.6, num=10)
        cond_l = hazus_conditional_liquefaction_probability(pga_l, "l")
        test_res = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.08057895,
                0.16852632,
                0.25647368,
                0.34442105,
                0.43236842,
                0.52031579,
                0.60826316,
                0.69621053,
                0.78415789,
                0.87210526,
                0.96005263,
                1.0,
            ]
        )

        np.testing.assert_array_almost_equal(cond_l, test_res)

    def test_hazus_conditional_liquefaction_probability_m(self):
        # Replicates Fig. 4.6 in the Hazus manual
        # However values do not match figure exactly, though
        # the formula and coefficients are double-checked...
        pga_m = np.linspace(0.1, 0.4, num=10)
        cond_m = hazus_conditional_liquefaction_probability(pga_m, "m")
        test_res = np.array(
            [
                0.0,
                0.0,
                0.11166667,
                0.334,
                0.55633333,
                0.77866667,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        )

        np.testing.assert_array_almost_equal(cond_m, test_res)

    def test_hazus_conditional_liquefaction_probability_h(self):
        # Replicates Fig. 4.6 in the Hazus manual
        # However values do not match figure exactly, though
        # the formula and coefficients are double-checked...
        pga_h = np.linspace(0.1, 0.3, num=10)
        cond_h = hazus_conditional_liquefaction_probability(pga_h, "h")
        test_res = np.array(
            [
                0.0,
                0.01744444,
                0.18788889,
                0.35833333,
                0.52877778,
                0.69922222,
                0.86966667,
                1.0,
                1.0,
                1.0,
            ]
        )

        np.testing.assert_array_almost_equal(cond_h, test_res)

    def test_hazus_conditional_liquefaction_probability_vh(self):
        # Replicates Fig. 4.6 in the Hazus manual
        # However values do not match figure exactly, though
        # the formula and coefficients are double-checked...
        pga_vh = np.linspace(0.05, 0.25, num=10)
        cond_vh = hazus_conditional_liquefaction_probability(pga_vh, "vh")
        test_res = np.array(
            [0.0, 0.0, 0.0385, 0.2405, 0.4425, 0.6445, 0.8465, 1.0, 1.0, 1.0]
        )

        np.testing.assert_array_almost_equal(cond_vh, test_res)
