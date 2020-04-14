from typing import Optional, Union

import numpy as np

from .liquefaction import fema_magnitude_correction_factor

def fema_lateral_spreading_displacements(mag, pga, pga_threshold, **kwargs):
    """

    """
    pga_ratio = pga / pga_threshold

    if np.isscalar(pga_ratio):
        if pga_ratio < 1.:
            a = 0.
        elif 1. < pga_ratio <=2:
            a = 12. * pga_ratio - 12.
        elif 2. < pga_ratio <= 3:
            a = 18. * pga_ratio - 24.
        #elif 3 < pga_ratio <= 4:
        else:
            a = 70. * pga_ratio - 180.

    else:
        a = np.zeros(pga_ratio.shape)
        a[1. < pga_ratio <= 2.] = 12. * pga_ratio[1. < pga_ratio <= 2.] - 12.
        a[2. < pga_ratio <= 3.] = 18. * pga_ratio[2. < pga_ratio <= 3.] - 24.
        a[3. < pga_ratio] = 70. * pga_ratio[3. < pga_ratio] - 180.

    mag_corr_factor = fema_magnitude_correction_factor(mag, **kwargs)

    return mag_corr_factor * a

        
def fema_displacement_correction_factor_spreading(mag:float, m3_coeff: float=0.0086,
    m2_coeff: float=-0.0914, m1_coeff: float=0.4698, intercept:float=0.9835
    )->float:
    """

    """
    return fema_magnitude_correction_factor(mag, m3_coeff=m3_coeff,
        m2_coeff=m2_coeff, m1_coeff=m1_coeff, intercept=intercept)
