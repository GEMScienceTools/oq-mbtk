#!/usr/bin/env python
# coding: utf-8

import os
import toml
import pandas as pd
import geopandas as gpd
import numpy as np

from glob import glob
from openquake.wkf.utils import create_folder, _get_src_id

import importlib
from openquake.baselib import sap
from openquake.hazardlib.sourcewriter import write_source_model
from openquake.hazardlib.source import PointSource
from openquake.hazardlib.mfd import TruncatedGRMFD

from shapely.geometry import Point as PointShapely
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.geo.nodalplane import NodalPlane
from openquake.hazardlib.tom import PoissonTOM


def _get_nodal_plane_distribution(data):
    out = []
    for tmp in data:
        out.append([tmp[0], NodalPlane(tmp[1], tmp[2], tmp[3])])
    return PMF(out)


def _get_hypocenter_distribution(data):
    out = []
    for tmp in data:
        out.append([np.around(float(tmp[0]), 2), np.around(float(tmp[1]), 2)])
    return PMF(out)


def create_nrml_sources(fname_input_pattern: str, fname_config: str, 
                        folder_out: str, fname_subzone_shp: str="", 
                        fname_subzone_config: str=""):

    create_folder(folder_out)
    
    # If true we take some of the information from subzones
    subzones = (len(fname_subzone_shp) > 0 and len(fname_subzone_config) > 0)
    if subzones:
        polygons_gdf = gpd.read_file(fname_subzone_shp)
        model_subz = toml.load(fname_subzone_config) 

    # This is used to instantiate the MSR
    module = importlib.import_module('openquake.hazardlib.scalerel')

    # Parsing config
    model = toml.load(fname_config)

    rms = model['rupture_mesh_spacing']
    mmin = model['mmin']
    bwid = model['bin_width']
    tom = PoissonTOM(1.0)

    # Processing files
    for fname in glob(fname_input_pattern):

        src_id = os.path.basename(fname).split('.')[0]
        #src_id = _get_src_id(fname)
        
        df = pd.read_csv(fname)
        
        # Create a geodataframe with the points in a given zone
        if subzones:

            # Create a geodataframe with points
            geom = [PointShapely(xy) for xy in zip(df.lon, df.lat)]
            gdf = gpd.GeoDataFrame(df, crs='epsg:4326', geometry=geom)

            # Select subzones within a zone
            tdf = polygons_gdf[polygons_gdf["parent"] == src_id]

            # Should contain the points within
            df = gpd.sjoin(gdf, tdf, op='within')

        # This is the information on the source in the config file
        srcd = model['sources'][src_id]

        # Looping over the points
        srcs = []
        for idx, pnt in df.iterrows():

            if subzones:
                srcd_sz = model_subz['sources'][pnt.id]

            sid = '{:s}_{:d}'.format(src_id, idx)
            name = ""

            trt = srcd['tectonic_region_type']
            msr_str = model['msr'][trt]

            my_class = getattr(module, msr_str)
            msr = my_class()

            # Get mmax and set the MFD
            mmx = srcd['mmax']
            mfd = TruncatedGRMFD(mmin, mmx, bwid, pnt.agr, pnt.bgr)

            key = 'rupture_aspect_ratio'
            rar = get_param(srcd, model['default'], key)

            key = 'upper_seismogenic_depth'
            usd = get_param(srcd, model['default'], key)

            key = 'lower_seismogenic_depth'
            lsd = get_param(srcd, model['default'], key)

            key = 'nodal_plane_distribution'
            tmp = get_param(srcd, model['default'], key)
            npd = _get_nodal_plane_distribution(tmp)

            key = 'hypocenter_distribution'
            tmp = get_param(srcd, model['default'], key)
            hyd = _get_hypocenter_distribution(tmp)

            if subzones:
                tmp = get_param(srcd_sz, model['default'], key)
                npd = _get_nodal_plane_distribution(tmp)

            loc = Point(pnt.lon, pnt.lat)
            src = PointSource(sid, name, trt, mfd, rms, msr, rar, tom,
                              usd, lsd, loc, npd, hyd)
            srcs.append(src)

        # Write output file
        fname_out = os.path.join(folder_out, 'src_{:s}.xml'.format(src_id))
        write_source_model(fname_out, srcs, 'Zone {:s}'.format(src_id))


def get_param(dct, dct_default, key):
    if key in dct:
        return dct[key]
    else:
        return dct_default[key]
    

create_nrml_sources.fname_input_pattern = "Pattern for input .csv files"
create_nrml_sources.fname_config = "Name of the configuration file"
create_nrml_sources.folder_out = "Name of the output folder"
create_nrml_sources.fname_subzone_shp = "Name of the shapefile with subzones"
create_nrml_sources.fname_subzone_config = "Name of config file for subzones"

if __name__ == '__main__':
    sap.run(create_nrml_sources)
