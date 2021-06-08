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
    """

    features = []
    print(pattern)
    for fname in glob.glob(pattern):
        print(fname)
        dat = np.loadtxt(fname)
        tmp = LineString([(x, y) for x, y in zip(dat[:, 0], dat[:, 1])])
        features.append(Feature(geometry=tmp))
    feature_collection = FeatureCollection(features)
    with open(fname_output, 'w') as f:
        dump(feature_collection, f)


descr = 'The pattern to the text files with the profiles'
main.labels = descr
descr = 'The name of the output .geojson file'
main.config_fname = descr

if __name__ == '__main__':
    sap.run(main)
