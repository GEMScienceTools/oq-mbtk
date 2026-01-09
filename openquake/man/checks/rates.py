import matplotlib.pyplot as plt

from openquake.man.model import read
from openquake.man.checks.plotting import plot_mfd_cumulative
from openquake.man.checks.mfd import get_total_mfd

from openquake.mbt.tools.mfd import (
    get_evenlyDiscretizedMFD_from_truncatedGRMFD)
from openquake.hazardlib.mfd.truncated_gr import TruncatedGRMFD
from openquake.hazardlib.source.non_parametric import \
    NonParametricSeismicSource


def get_mags_rates(source_model_fname: str, time_span: float):
    """
    This computes the total rate for non-paramteric source modelling the
    occurrence of a single magnitude value.

    :param str source_model_fname:
        The name of the xml shapefile
    :param float time_span:
        The time in years to which the probability of occurrence refers to
    :returns:
        A tuple with two floats. The magnitude modelled and the corresponding
        total annual rate of occurrence.
    """

    # Read the source_model
    src_model, _ = read(source_model_fname, False)

    # Process sources
    rate = 0.
    mag = None
    for src in src_model:
        if isinstance(src, NonParametricSeismicSource):
            for dat in src.data:
                rupture = dat[0]
                pmf = dat[1].data
                rate += pmf[1][0]
                if mag is None:
                    mag = rupture.mag
                else:
                    assert abs(mag-rupture.mag) < 1e-2
    return mag, rate/time_span


def mfd_from_xml(source_model_fname):
    """
    :param str source_model_fname:
        The name of the xml
    """
    #
    # read the source_model
    src_model, info = read(source_model_fname)
    #
    # compute total mfd sources
    return get_total_mfd(src_model)


