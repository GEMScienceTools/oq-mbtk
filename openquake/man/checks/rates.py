import matplotlib.pyplot as plt

from openquake.man.model import read
from openquake.man.checks.plotting import plot_mfd_cumulative
from openquake.man.checks.mfd import get_total_mfd

from openquake.mbt.oqt_project import OQtProject
from openquake.mbt.tools.mfd import (
    get_evenlyDiscretizedMFD_from_truncatedGRMFD)
from openquake.hazardlib.mfd.truncated_gr import TruncatedGRMFD
from openquake.hazardlib.source.non_parametric import \
    NonParametricSeismicSource


def get_mags_rates(source_model_fname, time_span):
    """
    :param str source_model_fname:
        The name of the xml
    :param float time_span:
        In years
    """

    # Read the source_model
    src_model, info = read(source_model_fname, False)

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
    return mag, rate


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


def xml_vs_mfd(source_id, source_model_fname, model_id,
               oqmbt_project_fname):
    """
    :param str source_id:
        The ID of the source to be analysed
    :param str source_model_fname:
        The name of the xml shapefile
    :param str model_id:
        The model ID
    """
    #
    # read the source_model
    src_model, info = read(source_model_fname)
    #
    # compute total mfd sources
    tmfd = get_total_mfd(src_model)
    #
    # read project
    oqtkp = OQtProject.load_from_file(oqmbt_project_fname)
    model_id = oqtkp.active_model_id
    model = oqtkp.models[model_id]
    #
    # get source mfd
    src = model.sources[source_id]
    mfd = src.mfd
    if isinstance(src.mfd, TruncatedGRMFD):
        mfd = get_evenlyDiscretizedMFD_from_truncatedGRMFD(mfd)
    #
    # compute total mfd sources
    plt.figure(figsize=(10, 8))
    plot_mfd_cumulative(tmfd)
    plot_mfd_cumulative(mfd, title=source_model_fname)
