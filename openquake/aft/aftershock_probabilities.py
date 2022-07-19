from typing import Optional

import h5py
import pandas as pd
import numpy as np

from openquake.hazardlib.mfd import TruncatedGRMFD

from openquake.hazardlib.source.rupture import BaseRupture


def get_aftershock_grmfd(
    rup,
    a_val: Optional[float] = None,
    b_val: float = 1.0,
    gr_min: float = 4.6,
    gr_max: float = 7.9,
    bin_width=0.2,
    c: float = 0.015,
    alpha: float = 1.0,
):

    if not a_val:
        a_val = get_a(rup.mag, c=c, alpha=alpha)

    mfd = TruncatedGRMFD(
        min_mag=gr_min,
        max_mag=gr_max,
        bin_width=bin_width,
        a_val=a_val,
        b_val=b_val,
    )

    return mfd


def num_aftershocks(Mmain, c=0.015, alpha=1.0):
    return np.int_(c * 10 ** (alpha * Mmain))


def get_a(main_mag, c=0.01, alpha=1.0):
    N_above_0 = num_aftershocks(main_mag, c=c, alpha=alpha)

    a = np.log10(N_above_0)
    return a


def get_source_counts(sources):
    source_counts = [s.count_ruptures() for s in sources]
    source_cum_counts = np.cumsum(source_counts)
    source_cum_start_counts = np.insert(source_cum_counts[:-1], [0], 0)
    source_count_starts = {
        s.source_id: source_cum_start_counts[i] for i, s in enumerate(sources)
    }

    return source_counts, source_cum_counts, source_count_starts


def get_aftershock_rup_rates(
    rup: BaseRupture,
    aft_df: pd.DataFrame,
    min_mag: float = 4.7,
    rup_id: Optional[int] = None,
    a_val: Optional[float] = None,
    b_val: float = 1.0,
    gr_min: float = 4.6,
    gr_max: float = 7.9,
    bin_width=0.2,
    c: float = 0.015,
    alpha: float = 1.0,
):

    if rup.mag < min_mag:
        return

    if not rup_id:
        rup_id = rup.rup_id

    mfd = get_aftershock_grmfd(
        rup,
        a_val=a_val,
        b_val=b_val,
        gr_min=gr_min,
        gr_max=gr_max,
        bin_width=bin_width,
        c=c,
        alpha=alpha,
    )

    occur_rates = mfd.get_annual_occurrence_rates()

    aft_df["dist_probs"] = np.exp(-aft_df.d)

    aft_probs = []

    for (mbin, bin_rate) in occur_rates:
        these_rups = aft_df[aft_df.mag == mbin]
        total_rates = these_rups.dist_probs.sum()

        if total_rates > 0.0:
            rate_coeff = bin_rate / total_rates
            adjusted_rates = (
                these_rups.dist_probs * rate_coeff
            ) * rup.occurrence_rate
            aft_probs.append(adjusted_rates)

    aft_probs = pd.concat(aft_probs)
    aft_probs.name = (rup.source, rup_id)
    return aft_probs


def get_rup(src_id, rup_id, rup_gdf, source_groups):
    return rup_gdf.iloc[source_groups.groups[src_id]].iloc[rup_id].rupture


RupDist2 = np.dtype([("r1", np.int32), ("r2", np.int64), ("d", np.single)])


def make_source_dist_df(s_id, rdists, source_count_starts):
    source_dist_list = []

    for s2, dists in rdists[s_id].items():
        s2_dist_mat = np.empty(dists.shape, dtype=RupDist2)
        s2_dist_mat["r1"] = dists["r1"]
        s2_dist_mat["r2"] = np.int64(dists["r2"]) + source_count_starts[s2]
        s2_dist_mat["d"] = dists["d"]

        source_dist_list.append(s2_dist_mat)

    source_dist_list = np.hstack(source_dist_list)

    source_df = pd.DataFrame(source_dist_list)

    return source_df


def fetch_rup_from_source_dist_groups(
    rup_id,
    source_dist_df,
    rup_groups,
    rup_df,
):
    rup_dist_df = source_dist_df.iloc[rup_groups.groups[rup_id]][
        ["r2", "d"]
    ].set_index("r2")
    rup_dist_df["mag"] = rup_df.iloc[rup_dist_df.index]["mag"]

    return rup_dist_df


def rupture_aftershock_rates_per_source(
    s_id,
    rdists,
    source_count_starts,
    rup_df,
    source_groups,
    r_on=1,
    ns=1,
    min_mag: float = 4.7,
    rup_id: Optional[int] = None,
    a_val: Optional[float] = None,
    b_val: float = 1.0,
    gr_min: float = 4.6,
    gr_max: float = 7.9,
    bin_width=0.2,
    c: float = 0.015,
    alpha: float = 1.0,
):

    source_rup_adjustments = []

    source_dist_df = make_source_dist_df(s_id, rdists, source_count_starts)
    rup_groups = source_dist_df.groupby("r1")

    source_rups = list(rup_groups.groups.keys())

    for ir, rup_id in enumerate(source_rups):
        rup = get_rup(s_id, rup_id, rup_df, source_groups)

        if rup.mag >= min_mag:

            aft_dist = fetch_rup_from_source_dist_groups(
                rup_id, source_dist_df, rup_groups, rup_df
            )

            ra = get_aftershock_rup_rates(
                rup,
                aft_dist,
                rup_id=rup_id,
                min_mag=min_mag,
                a_val=a_val,
                b_val=b_val,
                gr_min=gr_min,
                gr_max=gr_max,
                bin_width=bin_width,
                c=c,
                alpha=alpha,
            )
            if len(ra) != 0:
                source_rup_adjustments.append(ra)

        r_on += 1

    return source_rup_adjustments
