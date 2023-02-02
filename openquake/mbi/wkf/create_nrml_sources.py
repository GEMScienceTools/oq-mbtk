#!/usr/bin/env python
# coding: utf-8

import os
import toml
import pandas as pd
import geopandas as gpd
import numpy as np

from glob import glob
from openquake.wkf.utils import create_folder

import importlib
from openquake.baselib import sap
from openquake.hazardlib.sourcewriter import write_source_model
from openquake.hazardlib.source import PointSource, MultiPointSource
from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.hazardlib.mfd. multi_mfd import MultiMFD

from shapely.geometry import Point as PointShapely
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.mesh import Mesh
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


def write_as_multipoint_sources(df, model, src_id, module, subzones,
                                model_subz, mmin, bwid, rms, tom, folder_out):

    # We do not support subzones in this case
    assert subzones is False
    srcd = model['sources'][src_id]

    # Prefix
    pfx = model.get("source_prefix", "")
    pfx += "_" if len(pfx) else pfx

    # Looping over the points
    lons = []
    lats = []
    avals = []
    settings = False
    for idx, pnt in df.iterrows():

        # Get mmax and set the MFD
        mmx = srcd['mmax']
        avals.append(pnt.agr)

        lons.append(pnt.lon)
        lats.append(pnt.lat)

        if not settings:

            trt = srcd['tectonic_region_type']
            msr_str = model['msr'][trt]

            my_class = getattr(module, msr_str)
            msr = my_class()

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

    name = src_id
    mmfd = MultiMFD('truncGutenbergRichterMFD',
                    size=len(avals),
                    min_mag=[mmin],
                    max_mag=[mmx],
                    bin_width=[bwid],
                    b_val=[pnt.bgr],
                    a_val=avals)

    mesh = Mesh(np.array(lons), np.array(lats))
    out_src_id = pfx + src_id
    srcmp = MultiPointSource(out_src_id, name, trt, mmfd, msr, rar, usd, lsd,
                             npd, hyd, mesh, tom)

    # Write output file
    fname_out = os.path.join(folder_out, 'src_{:s}.xml'.format(src_id))
    write_source_model(fname_out, [srcmp], 'Zone {:s}'.format(src_id))


def write_as_set_point_sources(df, model, src_id, module, subzones,
                               model_subz, mmin, bwid, rms, tom, folder_out):
    """
    :param model:
        The dictionary with the configuration
    """

    srcd = model['sources'][src_id]

    # Looping over the points
    name = ""
    srcs = []
    for idx, pnt in df.iterrows():

        if subzones:
            srcd_sz = model_subz['sources'][pnt.id]

        pfx = model.get("source_prefix", "")
        pfx += "_" if len(pfx) else pfx
        sid = '{:s}{:s}_{:d}'.format(pfx, src_id, idx)

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
        src = PointSource(sid, name, trt, mfd, rms, msr, rar, tom, usd, lsd,
                          loc, npd, hyd)
        srcs.append(src)

    # Write output file
    fname_out = os.path.join(folder_out, 'src_{:s}.xml'.format(src_id))
    write_source_model(fname_out, srcs, 'Zone {:s}'.format(src_id))


def create_nrml_sources(fname_input_pattern: str, fname_config: str,
                        folder_out: str, as_multipoint: bool,
                        fname_subzone_shp: str = "",
                        fname_subzone_config: str = "",):

    create_folder(folder_out)

    # If true we take some of the information from subzones
    subzones = (len(fname_subzone_shp) > 0 and len(fname_subzone_config) > 0)
    model_subz = None
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

        if as_multipoint:
            write_as_multipoint_sources(df, model, src_id, module, subzones,
                                        model_subz, mmin, bwid, rms, tom,
                                        folder_out)
        else:
            write_as_set_point_sources(df, model, src_id, module, subzones,
                                       model_subz, mmin, bwid, rms, tom,
                                       folder_out)


def get_param(dct, dct_default, key):
    if key in dct:
        return dct[key]
    else:
        return dct_default[key]


def main(fname_input_pattern: str, fname_config: str, folder_out: str,
         as_multipoint: bool = False, fname_subzone_shp: str = "",
         fname_subzone_config: str = ""):
    """
    Creates nrml sources using the information in the configuration file
    """
    create_nrml_sources(fname_input_pattern, fname_config, folder_out,
                        as_multipoint, fname_subzone_shp, fname_subzone_config)


main.fname_input_pattern = "Pattern for input .csv files"
main.fname_config = "Name of the configuration file"
main.folder_out = "Name of the output folder"
msg = "If true creates a multipoint source otherwise a set of point sources"
main.as_multipoint = msg
main.fname_subzone_shp = "Name of the shapefile with subzones"
main.fname_subzone_config = "Name of config file for subzones"

if __name__ == '__main__':
    sap.run(create_nrml_sources)
