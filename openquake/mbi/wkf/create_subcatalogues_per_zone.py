#!/usr/bin/env python
# coding: utf-8

from openquake.baselib import sap
from openquake.wkf.catalogue import create_subcatalogues


def main(fname_polygons: str, fname_cat: str, folder_out: str, *,
         source_ids: list=[], idl: bool = False):
    """
    Given a file (e.g. a shapefile) with a set of polygons and an earthquake
    catalogue, it creates a set of .csv files each one containing the
    earthquakes inside each polygon.
    """
    if idl == False:
        _ = create_subcatalogues_nongeopd(fname_polygons, fname_cat, folder_out, source_ids)
    else:
        _ = create_subcatalogues(fname_polygons, fname_cat, folder_out, source_ids)


main.fname_polygons = 'Name of a shapefile with polygons'
main.fname_cat = 'Name of the .csv file with the catalog'
main.folder_out = 'Name of the output folder'
main.source_ids = 'IDs of sources to be considered'
main.idl = 'boolean indicating if source polygons cross the international datel'

if __name__ == '__main__':
    sap.run(main)
