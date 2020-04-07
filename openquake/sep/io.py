import numpy as np
import pandas as pd
import xarray as xr


def make_dataset_from_oq_gmfs(
    gmf_file=None,
    sitemesh_file=None,
    ruptures_file=None,
    events_file=None,
    ds_name=None,
) -> xr.Dataset:

    gmfs = pd.read_csv(gmf_file)
    rups = pd.read_csv(ruptures_file)
    events = pd.read_csv(events_file)
    sites = pd.read_csv(sitemesh_file)

    pga_ds = xr.Dataset(
        {
            "{}".format(e): make_pga_xr(
                e, gmfs=gmfs, sites=sites, rups=rups, events=events
            )
            for e in events["event_id"]
        }
    )

    if ds_name is not None:
        pga_ds["name"] = ds_name

    return pga_ds


def make_pga_xr(
    event_id, events=None, gmfs=None, sites=None, rups=None
) -> xr.DataArray:

    rup_id = events.loc[event_id]["rup_id"]

    event_gmf = gmfs.loc[gmfs.event_id == event_id].merge(
        sites, on="site_id", how="inner"
    )

    event_gmf.set_index(["lon", "lat"])

    event_pga = event_gmf["gmv_PGA"]

    pga = xr.DataArray.from_series(event_pga).transpose()

    pga.assign_attrs({"M": rups.loc[rup_id].mag})
    pga.name = "PGA (event {})".format(event_id)

    return pga


def make_dataframes_from_oq_gmfs(
    gmf_file=None, ruptures_file=None, events_file=None
):

    gmfs = pd.read_csv(gmf_file)
    rups = pd.read_csv(ruptures_file)
    events = pd.read_csv(events_file)

    events["mags"] = pd.merge(events, rups, on="rup_id")["mag"]

