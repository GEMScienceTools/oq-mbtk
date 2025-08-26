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
import numpy as np
import unittest
import tempfile
import toml
import matplotlib.pyplot as plt
import subprocess
import pandas as pd

HERE = os.path.dirname(__file__)
CODE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'subcatalogues')

class test_createSubcatalogue(unittest.TestCase):
    """ test the assignment of events to subcatalogues """
    
    def setUp(self):
        self.out_folder = tempfile.mkdtemp()
        
    def testCreateSubcatalogueSimple(self):
        """ simple case of one polygon loaded from geojson"""
       
        polygons =  os.path.join(DATA_PATH, 'test_poly.geojson')
        cat = os.path.join(DATA_PATH, 'simple_poly_test_cat.csv')
        subcatalogues_folder = self.out_folder

        code = os.path.join(CODE, 'mbi', 'wkf', 'create_subcatalogues_per_zone.py')
        fmt = '{:s} {:s} {:s} {:s}'
        cmd = fmt.format(code, polygons, cat, subcatalogues_folder)
        out = subprocess.call(cmd, shell=True)
 
        res = pd.read_csv(os.path.join(self.out_folder, 'subcatalogue_zone_1.csv'))
        assert(len(res) == 10)
        
    def testCreateSubcatalogue1src(self):
        """Test that this applies only to one zone"""
        
        polygons =  os.path.join(DATA_PATH, 'smooth_poly.geojson')
        cat = os.path.join(DATA_PATH, 'smooth_test.csv')
        subcatalogues_folder = self.out_folder

        code = os.path.join(CODE, 'mbi', 'wkf', 'create_subcatalogues_per_zone.py')
        fmt = '{:s} {:s} {:s} {:s}'
        cmd = fmt.format(code, polygons, cat, subcatalogues_folder)
        out = subprocess.call(cmd, shell=True)
        
        ## This should contain 3 events
        res = pd.read_csv(os.path.join(self.out_folder, 'subcatalogue_zone_1.csv'))
        assert(len(res) == 3)
        
    def testCreateSubcatalogueSmooth(self):
        """Test with three polygons from Atlantic MOR:
        	read 3 polygons from shapefile and split catalogue"""
        
        polygons =  os.path.join(DATA_PATH,  'smooth_poly.geojson')
        cat = os.path.join(DATA_PATH, 'smooth_test.csv')
        subcatalogues_folder = self.out_folder

        code = os.path.join(CODE, 'mbi', 'wkf', 'create_subcatalogues_per_zone.py')
        fmt = '{:s} {:s} {:s} {:s}'
        cmd = fmt.format(code, polygons, cat, subcatalogues_folder)
        out = subprocess.call(cmd, shell=True)
        
        ## This should contain 3 events
        res = pd.read_csv(os.path.join(self.out_folder, 'subcatalogue_zone_1.csv'))
        assert(len(res) == 3)
        ## These zones should contain one event each
        res2 = pd.read_csv(os.path.join(self.out_folder, 'subcatalogue_zone_2.csv'))
        res3 = pd.read_csv(os.path.join(self.out_folder, 'subcatalogue_zone_3.csv'))
        assert(len(res2) == len(res3))
        ## zone 4 should be empty
        
    def testCreateSubcataloguesIDL(self):
        """ Test for where polygon crosses the international date line (180 meridian or antimeridian)"""
        
        print("IDL test")
        polygons =  os.path.join(DATA_PATH, 'idl_poly.geojson')
        cat = os.path.join(DATA_PATH, 'idl_testcat.csv')
        subcatalogues_folder = self.out_folder

        code = os.path.join(CODE, 'mbi', 'wkf', 'create_subcatalogues_per_zone.py')
        fmt = '{:s} {:s} {:s} {:s}'
        cmd = fmt.format(code, polygons, cat, subcatalogues_folder)
        out = subprocess.call(cmd, shell=True)
        
        res = pd.read_csv(os.path.join(self.out_folder, 'subcatalogue_zone_0.csv'))
        print("international date line test contains ", len(res), "events")
        assert(len(res) == 4)
