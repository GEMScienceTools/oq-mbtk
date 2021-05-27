"""
:module:`openquake.man.checks.mfd`
"""

import copy
import openquake.mbt.tools.mfd as mfdt
from openquake.hazardlib.mfd import TruncatedGRMFD


def get_total_mfd(sources, trt=None):
    """
    :param list sources:
        A list of :class:`openquake.hazardlib.source.Source` instances
    :returns:
        A :class:`openquake.man.checks.mfd.EEvenlyDiscretizedMFD` instance
    """
    cnt = 0
    for src in sources:
        if ((trt is not None and trt == src.tectonic_region_type) or
                (trt is None)):
            mfd = src.mfd
            if isinstance(src.mfd, TruncatedGRMFD):
                mfd = mfdt.get_evenlyDiscretizedMFD_from_truncatedGRMFD(mfd)
            if cnt == 0:
                mfdall = copy.copy(mfd)
            else:
                mfdall.stack(mfd)
            cnt += 1
    return mfdall
