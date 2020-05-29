from typing import Optional, Union

import numpy as np

from .liquefaction import fema_magnitude_correction_factor


def fema_lateral_spreading_displacements(mag, pga, pga_threshold, **kwargs):
    """

    """
    pga_ratio = pga / pga_threshold

    if np.isscalar(pga_ratio):
        if pga_ratio < 1.0:
            a = 0.0
        elif 1.0 < pga_ratio <= 2:
            a = 12.0 * pga_ratio - 12.0
        elif 2.0 < pga_ratio <= 3:
            a = 18.0 * pga_ratio - 24.0
        # elif 3 < pga_ratio <= 4:
        else:
            a = 70.0 * pga_ratio - 180.0

    else:
        a = np.zeros(pga_ratio.shape)
        a[1.0 < pga_ratio <= 2.0] = (
            12.0 * pga_ratio[1.0 < pga_ratio <= 2.0] - 12.0
        )
        a[2.0 < pga_ratio <= 3.0] = (
            18.0 * pga_ratio[2.0 < pga_ratio <= 3.0] - 24.0
        )
        a[3.0 < pga_ratio] = 70.0 * pga_ratio[3.0 < pga_ratio] - 180.0

    mag_corr_factor = fema_magnitude_correction_factor(mag, **kwargs)

    return mag_corr_factor * a


def fema_disp_mag_correction_factor_spreading(
    mag: float,
    m3_coeff: float = 0.0086,
    m2_coeff: float = -0.0914,
    m1_coeff: float = 0.4698,
    intercept: float = 0.9835,
) -> float:
    """
    Improves estimates of lateral spreading during liquefaction based on the
    magnitude of an earthquake.
    """
    return fema_magnitude_correction_factor(
        mag,
        m3_coeff=m3_coeff,
        m2_coeff=m2_coeff,
        m1_coeff=m1_coeff,
        intercept=intercept,
    )
