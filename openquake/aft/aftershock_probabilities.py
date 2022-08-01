import time
import logging
from typing import Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from openquake.calculators.base import run_calc
from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.hazardlib.source.rupture import BaseRupture

# typing
from openquake.hazardlib.source import BaseSeismicSource

from openquake.aft.rupture_distances import (
    calc_rupture_adjacence_dict_all_sources,
    get_close_source_pairs,
)


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
    gr_min: float = 4.5,
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

    if np.abs(gr_min - occur_rates[0][0]) > 0.01:
        mag_diff = gr_min - occur_rates[0][0]
        occur_rates = [(occ[0] + mag_diff, occ[1]) for occ in occur_rates]

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
    gr_min: float = 4.5,
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


def prep_source_data(
    sources: Sequence[BaseSeismicSource], source_info=None
) -> Tuple[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]:
    """
    Creates a Pandas DataFrame and a Groupby object for all ruptures
    in a sequence of seismic sources.  The DataFrame has some additional
    information that is used during the pairwise rupture distance
    calculations.
    """
    big_rup_list = []
    rup_inds = []

    for i, source in enumerate(tqdm(sources, leave=False)):
        source_rup_inds = []
        if source_info is not None:
            rup_list = [
                r
                for r in tqdm(
                    source.iter_ruptures(),
                    total=source_info[i]["num_ruptures"],
                    leave=False,
                )
            ]
        else:
            rup_list = [r for r in source.iter_ruptures()]
        for j, r in enumerate(rup_list):
            # r.source = source.source_id
            r.source = i
            source_rup_inds.append((i, j))

        big_rup_list.extend(rup_list)
        rup_inds.extend(source_rup_inds)

    rup_df = pd.DataFrame(
        index=np.arange(len(big_rup_list)),
        data=big_rup_list,
        columns=["rupture"],
    )
    logging.info("\tadding rupture attributes")
    rup_df["source"] = [r.source for r in tqdm(rup_df["rupture"])]
    rup_df["mag"] = [r.mag for r in tqdm(rup_df["rupture"])]
    rup_df["xyz"] = [r.surface.mesh.xyz for r in tqdm(rup_df["rupture"])]
    rup_df["oq_rup_ind"] = rup_inds

    logging.info("\tgrouping ruptures by source")
    source_groups = rup_df.groupby("source")
    return rup_df, source_groups


def sources_from_job_ini(job_ini):

    calc = run_calc(
        job_ini, calculation_mode="preclassical", split_sources="false"
    )

    sources = calc.csm.get_sources()
    source_info = calc.datastore["source_info"][:]

    for i, source in enumerate(sources):
        source.source_id = i

    return sources, source_info


def get_aftershock_rupture_rates(
    job_ini, dist_constant=4.0, c=0.25, b_val=0.85, gr_max=7.5, min_mag=6.0
):

    t0 = time.time()
    logging.info("Getting sources from model")
    sources, source_info = sources_from_job_ini(job_ini)
    t1 = time.time()
    logging.info(f"\nDone in {(t1 - t0 ) / 60 :0.1} min")

    # breakpoint()

    logging.info("Calculating close source pairs")
    source_pairs = get_close_source_pairs(sources)
    t2 = time.time()
    logging.info(f"Done in { (t2 - t1) / 60 :0.2} min")
    logging.info(
        f"{len(source_pairs)} source pairs out of {len(sources)**2} possible"
    )

    logging.info("Prepping source data")
    rup_df, source_groups = prep_source_data(sources, source_info=source_info)
    t3 = time.time()
    logging.info(f"Done in { (t3-t2) / 60 :0.2} min")

    logging.info("Calculating rupture distances")
    rup_dists = calc_rupture_adjacence_dict_all_sources(
        source_pairs, rup_df, source_groups
    )
    t4 = time.time()
    logging.info(f"Done in {(t4-t3) / 60 :0.2} min")

    source_counts, source_cum_counts, source_count_starts = get_source_counts(
        sources
    )
    t5 = time.time()

    logging.info("Calculating aftershock rates per source")
    rup_adjustments = []
    r_on = 1
    for ns, source in enumerate(tqdm(sources)):
        rup_adjustments.extend(
            rupture_aftershock_rates_per_source(
                source.source_id,
                rup_dists,
                source_count_starts=source_count_starts,
                rup_df=rup_df,
                source_groups=source_groups,
                r_on=r_on,
                ns=ns,
                c=c,
                b_val=b_val,
                gr_max=gr_max,
                gr_min=rup_df.mag.min(),
            )
        )
        r_on = source_cum_counts[ns] + 1
    t6 = time.time()
    logging.info(f"Done in {(t6-t5) / 60 :0.2} min")

    logging.info("Concatenating results")
    rr = [r for r in rup_adjustments if len(r) != 0]
    t7 = time.time()
    rup_adj_df = pd.concat([pd.DataFrame(r) for r in rr], axis=1).fillna(0.0)
    t8 = time.time()

    rup_adjustments = rup_adj_df.sum(axis=1)
    oq_rup_index = rup_df.loc[rup_adjustments.index, "oq_rup_ind"]
    rup_adjustments.index = oq_rup_index
    t9 = time.time()

    logging.info(f"\nDone in {(t9-t0) / 60 :0.3} min")

    return rup_adjustments
