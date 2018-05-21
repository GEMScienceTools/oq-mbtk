"""
:module:`openquake.man.checks.mfd`
"""

import oqmbt.tools.mfd as mfdt

from openquake.hazardlib.mfd import TruncatedGRMFD


def get_total_mfd(sources, trt=None):
    """
    :param list sources:
        A list of :class:`openquake.hazardlib.source.Source` instances
    :returns:
        A :class:`openquake.man.checks.mfd.EEvenlyDiscretizedMFD` instance
    """
    #
    #
    mfdall = mfdt.EEvenlyDiscretizedMFD(5.1, 0.1, [1e-20])
    for src in sources:
        if ((trt is not None and trt == src.tectonic_region_type) or
                (trt is None)):
            mfd = src.mfd
            if isinstance(src.mfd, TruncatedGRMFD):
                mfd = mfdt.get_evenlyDiscretizedMFD_from_truncatedGRMFD(mfd)
            mfdall.stack(mfd)
    return mfdall
