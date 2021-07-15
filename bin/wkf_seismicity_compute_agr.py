#!/usr/bin/env python
# coding: utf-8

import toml
import geopandas as gpd
from openquake.baselib import sap


def calculate_agr(src_id: float, rate: float):
    """
    :param src_id:
    :param rate:
    """
    pass


def process(fname_config: str, fname_poly: str, use: str = [], skip: str = []):

    # Read config
    model = toml.load(fname_config)
    basel_mref_rate = model['baseline']['rate_mref']
    basel_mref = model['baseline']['mref']
    basel_bgr = model['baseline']['b_value']

    # Read polygons
    polygons_gdf = gpd.read_file(fname_poly)
    print(type(polygons_gdf))
    polygons_gdf = polygons_gdf.set_crs(4326, allow_override=True)

    # Loop over the polygons
    polygons_gdf.sort_values(by="id", ascending=True, inplace=True)
    polygons_gdf.reset_index(drop=True, inplace=True)

    for idx, poly in polygons_gdf.iterrows():

        src_id = poly.id
        if (len(use) and src_id not in use) or (src_id in skip):
            continue
        area = poly['geometry'].area
        print("Source ID", poly.id, area)


def main(fname_config: str, fname_poly: str, *, use: str = [], skip: str = []):
    process(fname_config, fname_poly, use, skip)


main.folder_name = "The name of the folder with smoothing results per source"
main.folder_name_out = "The name of the folder where to store the results"
main.fname_config = ".toml configuration file"
main.fname_poly = "The name of the shapefile with the polygons"
main.skip = "A string containing a list of source IDs"

if __name__ == '__main__':
    sap.run(main)
