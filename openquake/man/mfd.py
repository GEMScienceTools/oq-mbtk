import numpy

from oqmbt.tools.mfd import mag_to_mo
from openquake.hazardlib.mfd import TruncatedGRMFD, EvenlyDiscretizedMFD


def get_rates_within_m_range(mfd, mmint=0.0, mmaxt=11.0):
    """
    :parameter mfd:
    :parameter mmint:
    :parameter mmaxt:
    """
    rtes = numpy.array(mfd.get_annual_occurrence_rates())
    idx = numpy.nonzero((rtes[:, 0] > mmint) & (rtes[:, 0] < mmaxt))
    return sum(rtes[idx[0], 1])


def get_moment_from_mfd(mfd):
    """
    This computed the total scalar seismic moment released per year by a
    source

    :parameter mfd:
        An instance of openquake.hazardlib.mfd
    :returns:
        A float corresponding to the rate of scalar moment released
    """
    if isinstance(mfd, TruncatedGRMFD):
        return mfd._get_total_moment_rate()
    elif isinstance(mfd, EvenlyDiscretizedMFD):
        occ_list = mfd.get_annual_occurrence_rates()
        mo_tot = 0.0
        for occ in occ_list:
            mo_tot += occ[1] * mag_to_mo(occ[0])
    return mo_tot
