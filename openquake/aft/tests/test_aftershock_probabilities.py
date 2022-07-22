import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import pytest

from openquake.hazardlib.geo import Polygon, Point
from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.hazardlib.scalerel.wc1994 import WC1994
from openquake.hazardlib.source.area import AreaSource
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.geo.nodalplane import NodalPlane

from openquake.aft.rupture_distances import (
    get_close_source_pairs,
    calc_rupture_adjacence_dict_all_sources,
    prep_source_data,
)

from openquake.aft.aftershock_probabilities import (
    #get_aftershock_grmfd,
    #num_aftershocks,
    #get_a,
    get_source_counts,
    #get_aftershock_rup_rates,
    #get_rup,
    #RupDist2,
    #make_source_dist_df,
    #fetch_rup_from_source_dist_groups,
    rupture_aftershock_rates_per_source,
)

area_source_1 = AreaSource(
    source_id="s1",
    name="s1",
    tectonic_region_type="ActiveShallowCrust",
    mfd=TruncatedGRMFD(
        min_mag=4.6, max_mag=8.0, bin_width=0.2, a_val=1.0, b_val=1.0
    ),
    magnitude_scaling_relationship=WC1994(),
    rupture_aspect_ratio=1.0,
    temporal_occurrence_model=PoissonTOM,
    upper_seismogenic_depth=0.0,
    lower_seismogenic_depth=30.0,
    nodal_plane_distribution=PMF([(1.0, NodalPlane(0.0, 90, 180.0))]),
    hypocenter_distribution=PMF([(1.0, 15.0)]),
    polygon=Polygon(
        [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 0.0, 0.0),
            Point(1.0, 1.0, 0.0),
            Point(0.0, 1.0, 0),
            Point(0.0, 0.0, 0.0),
        ]
    ),
    area_discretization=15.0,
    rupture_mesh_spacing=5.0,
)

area_source_2 = AreaSource(
    source_id="s2",
    name="s2",
    tectonic_region_type="ActiveShallowCrust",
    mfd=TruncatedGRMFD(
        min_mag=4.6, max_mag=8.0, bin_width=0.2, a_val=1.0, b_val=1.0
    ),
    magnitude_scaling_relationship=WC1994(),
    rupture_aspect_ratio=1.0,
    temporal_occurrence_model=PoissonTOM,
    upper_seismogenic_depth=0.0,
    lower_seismogenic_depth=30.0,
    nodal_plane_distribution=PMF([(1.0, NodalPlane(0.0, 90, 180.0))]),
    hypocenter_distribution=PMF([(1.0, 15.0)]),
    polygon=Polygon(
        [
            Point(2.0, 0.0, 0.0),
            Point(2.0, -1.0, 0.0),
            Point(3.0, -1.0, 0.0),
            Point(3.0, 0.0, 0),
            Point(2.0, 0.0, 0.0),
        ]
    ),
    area_discretization=15.0,
    rupture_mesh_spacing=5.0,
)

area_source_3 = AreaSource(
    source_id="s3",
    name="s3",
    tectonic_region_type="ActiveShallowCrust",
    mfd=TruncatedGRMFD(
        min_mag=4.6, max_mag=8.0, bin_width=0.2, a_val=1.0, b_val=1.0
    ),
    magnitude_scaling_relationship=WC1994(),
    rupture_aspect_ratio=1.0,
    temporal_occurrence_model=PoissonTOM,
    upper_seismogenic_depth=0.0,
    lower_seismogenic_depth=30.0,
    nodal_plane_distribution=PMF([(1.0, NodalPlane(0.0, 90, 180.0))]),
    hypocenter_distribution=PMF([(1.0, 15.0)]),
    polygon=Polygon(
        [
            Point(4.0, 0.0, 0.0),
            Point(4.0, 1.0, 0.0),
            Point(5.0, 1.0, 0.0),
            Point(5.0, 0.0, 0),
            Point(4.0, 0.0, 0.0),
        ]
    ),
    area_discretization=15.0,
    rupture_mesh_spacing=5.0,
)


def test_num_aftershocks_1():

    pass


def test_get_aftershock_rup_adjustments():
    """
    Essentially, the workflow for the whole process
    """

    sources = [area_source_1, area_source_2, area_source_3]
    rup_df, source_groups = prep_source_data(sources)

    source_pairs = get_close_source_pairs(sources)

    rup_dists = calc_rupture_adjacence_dict_all_sources(
        source_pairs, rup_df, source_groups
    )

    source_counts, source_cum_counts, source_count_starts = get_source_counts(
        sources
    )

    rup_adjustments = []

    r_on = 1
    for ns, source in enumerate(sources):
        rup_adjustments.extend(
            rupture_aftershock_rates_per_source(
                source.source_id,
                rup_dists,
                source_count_starts=source_count_starts,
                rup_df=rup_df,
                source_groups=source_groups,
                r_on=r_on,
                ns=ns,
                c=0.25,
                b_val=0.85,
                gr_max=7.5,
            )
        )
        r_on = source_cum_counts[ns] + 1

    rr = [r for r in rup_adjustments if len(r) != 0]

    rup_adj_df = pd.concat([pd.DataFrame(r) for r in rr], axis=1).fillna(0.0)

    rup_adjustments = rup_adj_df.sum(axis=1)

    return rup_adjustments


def mag_to_mo(mag: float, c: float = 9.05):
    """
    Scalar moment [in Nm] from moment magnitude
    :return:
        The computed scalar seismic moment
    """
    return 10 ** (1.5 * mag + c)


def plot_mfds():
    sources = [area_source_1, area_source_2, area_source_3]
    rup_df, source_groups = prep_source_data(sources)

    rup_adjustments = get_aftershock_rup_adjustments()

    rup_df["rates"] = [rup.occurrence_rate for rup in rup_df.rupture]

    aft_rup_rates = pd.Series(index=rup_df.index, data=np.zeros(len(rup_df)))
    aft_rup_rates = aft_rup_rates.add(rup_adjustments, fill_value=0.0)

    rates_w_aftershocks = rup_df.rates + aft_rup_rates

    print(rates_w_aftershocks.describe())

    mag_arg_sort = np.argsort(rup_df["mag"])[::-1]

    mag_sort = rup_df["mag"].values[mag_arg_sort]

    cum_rates = np.cumsum(rup_df.rates.values[mag_arg_sort])
    cum_aft_rates = np.cumsum(rates_w_aftershocks.values[mag_arg_sort])

    print(cum_rates[0], cum_rates[-1])
    print(cum_aft_rates[0], cum_aft_rates[-1])

    plt.figure()

    plt.semilogy(mag_sort, cum_rates, label="no aft")
    plt.semilogy(mag_sort, cum_aft_rates, label="aft", linestyle="-.")
    plt.legend()
    plt.show()

    return


def look_at_aftershock_rup_rates():
    sources = [area_source_1, area_source_2, area_source_3]
    rup_df, source_groups = prep_source_data(sources)
    source_pairs = get_close_source_pairs(sources)

    rup_dists = calc_rupture_adjacence_dict_all_sources(
        source_pairs, rup_df, source_groups
    )

    source_counts, source_cum_counts, source_count_starts = get_source_counts(
        sources
    )


#plot_mfds()
