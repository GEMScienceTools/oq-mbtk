"""
:module:`openquake.man.checks.mfd`
"""

import numpy as np
import openquake.mbt.tools.mfd as mfdt

from openquake.hazardlib.mfd import TruncatedGRMFD, ArbitraryMFD, \
    EvenlyDiscretizedMFD
from openquake.hazardlib.mfd.multi_mfd import MultiMFD
from openquake.hazardlib.source.non_parametric import \
    NonParametricSeismicSource


def get_total_mfd(sources, trt=None, bin_width=0.1):
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
            # Non parametric source
            if isinstance(src, NonParametricSeismicSource):
                dat = []
                for rup, pmf in src.data:
                    rte = - np.log(1-pmf.data[0, 0])
                    dat.append([rup.mag, rte])
                dat = np.array(dat)
                mmin = np.floor(min(dat[:, 0])/bin_width)*bin_width
                mmax = np.ceil(max(dat[:, 0])/bin_width)*bin_width
                edges = np.arange(mmin-bin_width/2, mmax+bin_width/1.99,
                                  bin_width)
                count, edges = np.histogram(dat[:, 0], edges,
                                            weights=dat[:, 1])
                mfd = EvenlyDiscretizedMFD(mmin, bin_width, count)
            elif isinstance(mfd, ArbitraryMFD):
                mfd = src.mfd
                mfd = mfdt.get_evenlyDiscretizedMFD_from_arbitraryMFD(
                    mfd, bin_width)
            # Truncated MFD
            elif isinstance(mfd, TruncatedGRMFD):
                mfd = src.mfd
                mfd = mfdt.get_evenlyDiscretizedMFD_from_truncatedGRMFD(
                    mfd, bin_width)
            # Multi MFD
            elif isinstance(mfd, MultiMFD):
                mfd = mfdt.EEvenlyDiscretizedMFD(5.1, 0.1, [1e-20])
                for tmp in src.mfd:
                    if isinstance(tmp, TruncatedGRMFD):
                        m = mfdt.get_evenlyDiscretizedMFD_from_truncatedGRMFD(
                            tmp, bin_width)
                        mfd.stack(m)
                    elif isinstance(tmp, ArbitraryMFD):
                        m = mfdt.get_evenlyDiscretizedMFD_from_arbitraryMFD(
                            tmp, bin_width)
                        mfd.stack(m)
                    elif isinstance(tmp, EvenlyDiscretizedMFD):
                        mfd.stack(tmp)
                    else:
                        print(tmp)
                        raise ValueError('Unsupported MFD')
            mfdall.stack(mfd)
    return mfdall
