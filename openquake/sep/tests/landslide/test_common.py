import unittest

import numpy as np

from openquake.sep.landslide.common import static_factor_of_safety

slope_array = np.linspace(0.0, 60.0)


def test_static_factor_of_safety_dry():
    sfs = static_factor_of_safety(
        slope_array, cohesion=20e3, friction_angle=30.0
    )

    sfs_ = np.array(
        [
            6.20240095e06,
            5.06510388e01,
            2.53226659e01,
            1.68786065e01,
            1.26556254e01,
            1.01210752e01,
            8.43074034e00,
            7.22281393e00,
            6.31639255e00,
            5.61097425e00,
            5.04625782e00,
            4.58386974e00,
            4.19822763e00,
            3.87162059e00,
            3.59139796e00,
            3.34828255e00,
            3.13531646e00,
            2.94717892e00,
            2.77973153e00,
            2.62970712e00,
            2.49449194e00,
            2.37197006e00,
            2.26041028e00,
            2.15838244e00,
            2.06469473e00,
            1.97834603e00,
            1.89848920e00,
            1.82440261e00,
            1.75546769e00,
            1.69115120e00,
            1.63099097e00,
            1.57458445e00,
            1.52157942e00,
            1.47166631e00,
            1.42457198e00,
            1.38005450e00,
            1.33789879e00,
            1.29791303e00,
            1.25992557e00,
            1.22378236e00,
            1.18934475e00,
            1.15648761e00,
            1.12509770e00,
            1.09507231e00,
            1.06631808e00,
            1.03874991e00,
            1.01229011e00,
            9.86867591e-01,
            9.62417171e-01,
            9.38878988e-01,
        ]
    )

    np.testing.assert_allclose(sfs, sfs_, rtol=1e-4)


test_static_factor_of_safety_dry()
