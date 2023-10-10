#!/usr/bin/env python
# coding: utf-8

import pathlib
import unittest

from openquake.fnm.importer import kite_surfaces_from_geojson

HERE = pathlib.Path(__file__).parent
PLOTTING = False

class TestImportGeojson(unittest.TestCase):

    def test_get_surfs(self):
        fname = HERE / 'data' / 'kunlun_faults.geojson'
        # Import 21 faults
        surfs = kite_surfaces_from_geojson(fname, 5)

        if PLOTTING:
            from openquake.fnm.plot import plot
            meshes = [s.mesh for s in surfs]
            plot(meshes)



