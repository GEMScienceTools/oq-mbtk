#!/usr/bin/env python
# coding: utf-8

import re
import os
import toml
import pygmt
import json
import numpy as np
import pandas as pd
import geojson as geoj
import geopandas as gpd
import matplotlib.pyplot as plt
from openquake.baselib import sap
from shapely.geometry import shape
from openquake.wkf.utils import create_folder
from openquake.commonlib.datastore import read
from openquake.wkf.compute_gr_params import get_mmax_ctab
from openquake.hmtk.seismicity.catalogue import Catalogue
from openquake.mbt.tools.model_building.plt_tools import _load_catalogue
from openquake.hmtk.seismicity.occurrence.utils import get_completeness_counts


def from_df(df, end_year=None):
    cat = Catalogue()
    for column in df:
        if (column in Catalogue.FLOAT_ATTRIBUTE_LIST or
                column in Catalogue.INT_ATTRIBUTE_LIST):
            cat.data[column] = df[column].to_numpy()
        else:
            cat.data[column] = df[column]
    cat.end_year = np.max(df.year) if end_year is None else end_year
    return cat


def to_df(cat):
    df = pd.DataFrame()
    for key in cat.data:
        if key not in ['comment', 'flag']:
            df.loc[:, key] = cat.data[key]
    return df


def print_example():
    tmps = """
[main]
# Description of the check
description = "Puerto Rico"
# ID of the calculation containing the SES
calc_id = 1180
# Duration in years
ses_duration = 10000
# Width of the bins in the MFD used for the comparison
bin_width = 0.5
# Minimum magnitude used to compare the two datasets
min_magnitude = 5.0
# Catalogues containing the observed seismicity
catalogues = ["./../../catalogueA",
              "./../../catalogueB"]
# Name of the file with polygon defining the investigation area. It must
# be a multipolygon layer
polygon = "./../polygons/prc.geojson"
# Directory where to store the results
output_dir = "./../output/prc"

# Name of the .toml file used to build the model
fname_config = "./../../subduction/config/m01_int_config.toml"
completeness_label = "lan"

OR

# Completeness table
completeness_table = [ [ 2000.0, 4.5,], [ 1970.0, 5.0,], [ 1940.0, 6.0,],
                       [ 1918.0, 7.0,], [ 1900.0, 7.7,],]
    """
    print(tmps)


def check_ses_vs_catalogue(fname: str, *, example_flag: bool = False):
    """ Compares SES against a catalogue given a .toml configuration file """

    # Print an example of configuration file
    if example_flag:
        print_example()
        exit()

    # Load the .toml file containing the information required
    config_main = toml.load(fname)
    path = os.path.dirname(fname)

    print('Root path: {:s}'.format(path))

    # Read information in the config file
    fname_catalogues = []
    for tmp_name in config_main['main']['catalogues']:
        # If not absolute
        if not re.search('^/', tmp_name):
            tmp_name = os.path.join(path, tmp_name)
            assert os.path.exists(tmp_name)
            print('Catalogue: {:s}'.format(tmp_name))
        fname_catalogues.append(tmp_name)
    calc_id = config_main['main']['calc_id']
    ses_duration = config_main['main']['ses_duration']
    polygon_fname = os.path.join(path, config_main['main']['polygon'])
    output_dir = os.path.join(path, config_main['main']['output_dir'])
    descr = config_main['main']['description']
    binw = config_main['main'].get('bin_width', 0.2)
    min_magnitude = config_main['main'].get('min_magnitude', None)

    if ('tectonic_region' not in config_main['main'] or
            config_main['main']['tectonic_region'] in ['', 'none', 'None']):
        tectonic_region = None
    else:
        tectonic_region = int(config_main['main']['tectonic_region'])

    # Checking
    msg = 'The polygon file does not exist:\n{:s}'.format(polygon_fname)
    assert os.path.exists(polygon_fname), msg
    if not os.path.exists(output_dir):
        create_folder(output_dir)

    # Reading ruptures from the datastore
    dstore = read(calc_id)
    dfr = dstore.read_df('ruptures')
    dfr = gpd.GeoDataFrame(dfr, geometry=gpd.points_from_xy(dfr.hypo_0,
                                                            dfr.hypo_1))
    if tectonic_region is not None:
        dfr = dfr.loc[dfr['trt_smr'] == tectonic_region]

    # Reading geojson polygon and create the shapely geometry
    with open(polygon_fname) as json_file:
        data = json.load(json_file)
    polygon = data['features'][0]['geometry']
    tmp = eval(geoj.dumps(polygon))
    geom = shape(tmp)

    # Get region limits
    coo = []
    for poly in geom.geoms:
        coo += list(zip(*poly.exterior.coords.xy))
    coo = np.array(coo)
    minlo = np.min(coo[:, 0])
    minla = np.min(coo[:, 1])
    maxlo = np.max(coo[:, 0])
    maxla = np.max(coo[:, 1])
    region = "{:f}/{:f}/{:f}/{:f}".format(minlo, maxlo, minla, maxla)

    # Read catalogue
    for i, fname in enumerate(fname_catalogues):
        if i == 0:
            tcat = _load_catalogue(fname)
        else:
            tcat.concatenate(_load_catalogue(fname))

    # Create a dataframe from the catalogue
    dfcat = to_df(tcat)
    dfcat = gpd.GeoDataFrame(dfcat,
                             geometry=gpd.points_from_xy(dfcat.longitude,
                                                         dfcat.latitude))
    dfcat.head(n=1)

    # Select the events within the polygon and convert from df to catalogue
    idx = dfcat.within(geom)
    selcat_df = dfcat.loc[idx]
    selcat = from_df(selcat_df)

    if 'completeness_table' in config_main['main']:
        ctab = config_main['main']['completeness_table']
        ctab = np.array(ctab)
    else:
        fname_config = os.path.join(path, config_main['main']['fname_config'])
        msg = 'The config file does not exist:\n{:s}'.format(fname_config)
        assert os.path.exists(fname_config), msg
        config = toml.load(fname_config)
        completeness_label = config_main['main']['completeness_label']
        _, ctab = get_mmax_ctab(config, completeness_label)

    if len(selcat_df.magnitude) < 2:
        print('The catalogue contains less than 2 earthquakes')
        return

    selcat.data["dtime"] = selcat.get_decimal_time()
    cent_mag, t_per, n_obs = get_completeness_counts(selcat, ctab, binw)
    tmp = n_obs/t_per
    hiscml_cat = np.array([np.sum(tmp[i:]) for i in range(0, len(tmp))])

    # Take into account possible multiple occurrences in the SES
    df = dfr.loc[dfr.index.repeat(dfr.n_occ)]
    assert len(df) == np.sum(dfr.n_occ)

    # SES histogram
    idx = dfr.within(geom)
    bins = np.arange(min_magnitude, 9.0, binw)
    hisr, _ = np.histogram(df.loc[idx].mag, bins=bins)
    hisr = hisr / ses_duration
    hiscml = np.array([np.sum(hisr[i:]) for i in range(0, len(hisr))])

    # Plotting
    fig = plt.figure(figsize=(7, 5))
    # - cumulative
    plt.plot(bins[:-1], hiscml, '--x', label='SES')
    plt.plot(cent_mag-binw/2, hiscml_cat, '-.x', label='Catalogue')
    # - incremental
    plt.bar(cent_mag, n_obs/t_per, width=binw*0.7, fc='none', ec='red',
            alpha=0.5, align='center')
    plt.bar(bins[1:]-binw/2, hisr, width=binw*0.6, fc='none', ec='blue',
            alpha=0.5)
    plt.yscale('log')
    _ = plt.xlabel('Magnitude')
    _ = plt.ylabel('Annual frequency of exceedance')
    plt.grid()
    plt.legend()
    plt.title(descr)
    # - set xlim
    xlim = list(fig.gca().get_xlim())
    xlim[0] = min_magnitude if min_magnitude is not None else xlim[0]
    plt.xlim(xlim)
    plt.savefig(os.path.join(output_dir, 'ses.png'))

    # Plot map with the SES
    fig = pygmt.Figure()
    fig.basemap(region=region, projection="M15c", frame=True)
    fig.coast(land="#666666", water="skyblue")
    pygmt.makecpt(cmap="jet", series=[0, 300])
    fig.plot(x=dfr.loc[idx].hypo_0,
             y=dfr.loc[idx].hypo_1,
             style="c",
             color=dfr.loc[idx].hypo_2,
             cmap=True,
             size=0.01 * (1.5 ** dfr.loc[idx].mag),
             pen="black")
    fig.show()
    fig.savefig(os.path.join(output_dir, 'map_ses.png'))

    # Plot map with catalogue
    fig = pygmt.Figure()
    fig.basemap(region=region, projection="M15c", frame=True)
    fig.coast(land="#666666", water="skyblue")
    pygmt.makecpt(cmap="jet", series=[0, 300])
    fig.plot(x=selcat_df.longitude,
             y=selcat_df.latitude,
             style="c",
             color=selcat_df.depth,
             cmap=True,
             size=0.01 * (1.5 ** selcat_df.magnitude),
             pen="black")
    fig.show()
    fig.savefig(os.path.join(output_dir, 'map_eqks.png'))

    # Depth histogram
    deptw = 10.
    mmin = 5.0
    dfs = df.loc[idx]
    bins = np.arange(0.0, 200.0, deptw)
    fig = plt.figure()
    hisr, _ = np.histogram(dfs[dfs.mag > mmin].hypo_2, bins=bins)
    hiscat, _ = np.histogram(selcat_df[selcat_df.magnitude > mmin].depth,
                             bins=bins)
    fig = plt.Figure(figsize=(5, 8))
    plt.barh(bins[:-1], hisr/sum(hisr), align='edge', height=deptw*0.6,
             fc='lightgreen', ec='blue', label='ses')
    plt.barh(bins[:-1], hiscat/sum(hiscat), align='edge', height=deptw*0.5,
             fc='white', ec='red', alpha=0.5, lw=1.5, label='catalogue')
    for dep, val in zip(bins[:-1], hiscat):
        if val > 0:
            plt.text(val/sum(hiscat), dep, s='{:.2f}'.format(val))
    plt.gca().invert_yaxis()
    _ = plt.ylabel('Depth [km]')
    _ = plt.xlabel('Count')
    plt.grid()
    plt.legend()
    plt.title(descr)
    plt.savefig(os.path.join(output_dir, 'depth_normalized.png'))
