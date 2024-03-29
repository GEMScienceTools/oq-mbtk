#!/usr/bin/env python
# coding: utf-8

import glob
import numpy as np
from geojson import LineString, FeatureCollection, dump, Feature
from openquake.baselib import sap


def main(pattern: str, fname_output: str = "profiles.geojson"):
    """
    Creates a geojson file with all the sections included in the text files
    matching the pattern

    Example use: 
    oqm sub geojson_from_profiles 'openquake/sub/tests/data/cs_cam/cs*csv'
    """

    features = []
    for fname in glob.glob(pattern):
        dat = np.loadtxt(fname)
        tmp = LineString([(x, y) for x, y in zip(dat[:, 0], dat[:, 1])])
        prop = {'csid': fname.split('_')[1].replace('.csv','')}
        features.append(Feature(geometry=tmp, properties=prop))
    feature_collection = FeatureCollection(features)
    with open(fname_output, 'w') as f:
        dump(feature_collection, f)
    print(f'profiles written to {fname_output}')


descr = 'The pattern to the text files with the profiles'
main.labels = descr
descr = 'The name of the output .geojson file'
main.config_fname = descr

if __name__ == '__main__':
    sap.run(main)
