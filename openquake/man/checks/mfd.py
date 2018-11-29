"""
:module:`openquake.man.checks.mfd`
"""

import openquake.mbt.tools.mfd as mfdt

from openquake.hazardlib.mfd import TruncatedGRMFD, ArbitraryMFD


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
        print(src.source_id)
        if ((trt is not None and trt == src.tectonic_region_type) or
                (trt is None)):
            mfd = src.mfd
            if isinstance(src.mfd, ArbitraryMFD):
                mfd = mfdt.get_evenlyDiscretizedMFD_from_arbitraryMFD(mfd)
            elif isinstance(src.mfd, TruncatedGRMFD):
                mfd = mfdt.get_evenlyDiscretizedMFD_from_truncatedGRMFD(mfd)
            mfdall.stack(mfd)
    return mfdall
