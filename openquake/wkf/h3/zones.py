#!/usr/bin/env python
# coding: utf-8

import os
import h3
import json
import shapely
import geopandas as gpd
from openquake.wkf.utils import create_folder


def discretize_zones_with_h3_grid(h3_level: str, fname_poly: str,
                                  folder_out: str):

    h3_level = int(h3_level)
    create_folder(folder_out)

    tmp = "mapping_h{:d}.csv".format(h3_level)
    fname_out = os.path.join(folder_out, tmp)

    # Read polygons
    polygons_gdf = gpd.read_file(fname_poly)

    # Select point in polygon
    fout = open(fname_out, 'w')
    for idx, poly in polygons_gdf.iterrows():

        geojson_poly = eval(json.dumps(shapely.geometry.mapping(poly.geometry)))

        # Revert the positions of lons and lats
        coo = [[c[1], c[0]] for c in geojson_poly['coordinates'][0]]
        geojson_poly['coordinates'] = [coo]

        # Discretizing
        hexagons = list(h3.polyfill(geojson_poly, h3_level))
        for hxg in hexagons:
            if isinstance(poly.id, str):
                fout.write("{:s},{:s}\n".format(hxg, poly.id))
            else:
                fout.write("{:s},{:d}\n".format(hxg, poly.id))

    fout.close()
