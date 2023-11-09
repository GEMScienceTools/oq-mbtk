# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

import os
import subprocess
import pandas as pd
import numpy as np
import unittest
import tempfile
import toml
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'subcatalogues')
CODE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class test_create_subcatalogue(unittest.TestCase):
    """ test the assignment of events to subcatalogues """
    
    def setUp(self):
        self.out_folder = tempfile.mkdtemp()
        #self.out_folder = "~/tests/test_out"
    def test_create_subcatalogue_simple(self):
        """ simple case of one polygon loaded from geojson"""
        
        print("simple test")
        polygons =  os.path.join(DATA_PATH, 'test_poly.geojson')
        cat = os.path.join(DATA_PATH, 'simple_poly_test_cat.csv')
        subcatalogues_folder = self.out_folder
        
        print(cat)
        #code = os.path.join(CODE, 'mbi', 'wkf', 'create_subcatalogues_per_zone.py')
        #fmt = '{:s} {:s} {:s} {:s}'
        #cmd = fmt.format(code, polygons, cat, subcatalogues_folder)
        #out = subprocess.call(cmd, shell=True)

        code = os.path.join(HERE, 'create_subcatalogues_per_zone.py')
        print(code)
        fmt = '{:s} {:s} {:s} {:s}'
        cmd = fmt.format(code, polygons, cat, subcatalogues_folder)
        out = subprocess.call(cmd, shell=True)
        #assert out.returncode == 0
        
        out_fname = os.path.join(self.out_folder, 'subcatalogue_zone_1.csv')
        print(out_fname)
        res = pd.read_csv(out_fname)
        print("simple cataloue contains ", len(res), "events")
        assert(len(res) == 10)
        
    def test_create_subcatalogue_smooth(self):
        """Test with data from Atlantic MOR, 3 polygons from shapefile"""
        
        print("smooth test")
        polygons =  os.path.join(DATA_PATH, 'smooth_poly',  'smooth_test_poly.shp')
        cat = os.path.join(DATA_PATH, 'smooth_test.csv')
        subcatalogues_folder = self.out_folder

        code = os.path.join(CODE, 'mbi', 'wkf', 'create_subcatalogues_per_zone.py')
        #code = os.path.join(HERE, 'create_subcatalogues_per_zone')
        fmt = '{:s} {:s} {:s} {:s}'
        cmd = fmt.format(code, polygons, cat, subcatalogues_folder)
        out = subprocess.call(cmd, shell=True)
        #assert out.returncode == 0
        
        ## This should contain 3 events
        res = pd.read_csv(os.path.join(self.out_folder, 'subcatalogue_zone_1.csv'))
        assert(len(res) == 3)
        ## These zones should contain one event each
        res2 = pd.read_csv(os.path.join(self.out_folder, 'subcatalogue_zone_2.csv'))
        res3 = pd.read_csv(os.path.join(self.out_folder, 'subcatalogue_zone_3.csv'))
        assert(len(res2) == len(res3))
        print("zone 2 contains ", len(res2), " events")
        ## zone 4 should be empty
        
    def test_create_subcatalogues_IDL(self):
        """ Test for where polygon crosses the international date line (180 meridian or antimeridian)"""
        
        print("IDL test")
        polygons =  os.path.join(DATA_PATH, 'idl_poly.geojson')
        cat = os.path.join(DATA_PATH, 'idl_testcat.csv')
        subcatalogues_folder = self.out_folder

        code = os.path.join(CODE, 'mbi', 'wkf', 'create_subcatalogues_per_zone.py')
        #code = os.path.join(HERE, 'create_subcatalogues_per_zone')
        fmt = '{:s} {:s} {:s} {:s}'
        cmd = fmt.format(code, polygons, cat, subcatalogues_folder)
        # diff between subprocess.run and subprocess.call
        out = subprocess.call(cmd, shell=True)
        #assert out.returncode == 0
               
        res = pd.read_csv(os.path.join(self.out_folder, 'subcatalogue_zone_1.csv'))
        print("international date line test contains ", len(res), "events")
        assert(len(res) == 3)
