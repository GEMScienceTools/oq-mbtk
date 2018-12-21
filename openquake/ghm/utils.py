"""
Module :module:`~openquake.ghm.utils`
"""

import re
import geopandas as gpd

from shapely.geometry import Polygon, MultiPolygon


def explode(indf):
    """
    Implements what's suggested here: http://goo.gl/nrRpdV

    :param indf:
        A geodataframe instance
    :returns:
        A geodataframe instance
    """
    outdf = gpd.GeoDataFrame(columns=indf.columns)
    for idx, row in indf.iterrows():
        if type(row.geometry) == Polygon:
            outdf = outdf.append(row, ignore_index=True)
        if type(row.geometry) == MultiPolygon:
            multdf = gpd.GeoDataFrame(columns=indf.columns)
            recs = len(row.geometry)
            multdf = multdf.append([row]*recs, ignore_index=True)
            for geom in range(recs):
                multdf.loc[geom, 'geometry'] = row.geometry[geom]
            outdf = outdf.append(multdf, ignore_index=True)
    return outdf


def read_hazard_map_csv(fname):
    """
    Read the content of a .csv file with the mean hazard map computed for a
    given hazard model.

    :param str fname:
        The name of the file containing the results
    :return:
        A dictionary with key the sting in the header and values the floats
        in the csv file
    """
    data = {}
    for line in open(fname):
        if re.search('^#', line):
            pass
        else:
            if re.search('lon', line):
                labels = re.split('\,', line)
            else:
                aa = re.split('\,', line)
                for l, d in zip(labels, aa):
                    if l in data:
                        data[l].append(float(d))
                    else:
                        data[l] = [float(d)]
    return data


def create_query(inpt, field, labels):
    """
    Creates a query

    :param inpt:
    :param field:
    :param labels:
    :returns:
    """
    sel = None
    for lab in labels:
        if sel is None:
            sel = inpt[field] == lab
        else:
            sel = sel | (inpt[field] == lab)
    return inpt.loc[sel]
