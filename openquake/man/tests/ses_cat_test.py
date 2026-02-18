# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
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
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import os
import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil

from openquake.man.ses_cat import (
    merge_ses_event_rupture, 
    add_random_datetime, 
    convert_to_hmtk,
    build_hmtk_ses_catalogue
)

EVENTS_TEXT = """event_id,rup_id,year,ses_id,source_id,rlz_id
0,118,5948,1,fr,0
1,65,3266,12,fr,1
2,91,4567,18,fr,2
3,136,6815,24,fr,3
4,11,554,121,fr,4
5,195,9763,123,fr,5
6,142,7128,214,fr,6
7,177,8881,349,fr,7
8,150,7515,504,fr,8
9,88,4425,504,fr,9
"""

RUPTURES_TEXT = """rup_id,centroid_lon,centroid_lat,mag,centroid_depth,multiplicity,trt,strike,dip,rake
118,4.5528,44.82738,3.55,5,1,Active Shallow Crust,0,89.99,0
65,4.5528,44.82738,3.65,5,1,Active Shallow Crust,0,89.98,0
91,4.5528,44.82738,3.65,5,1,Active Shallow Crust,0,89.98,-90
136,4.5528,44.82738,3.75,5,1,Active Shallow Crust,0,89.98,0
11,4.5528,44.82738,4.55,15,1,Active Shallow Crust,0,89.99,0
195,4.5528,44.82738,4.55,5,1,Active Shallow Crust,240.0,57.99,0
142,4.5528,44.82738,5.25,15,1,Active Shallow Crust,128.8,89.98,0
177,4.5528,44.82738,6.45,15,1,Active Shallow Crust,0,89.99,0
150,-5.48151,49.46184,3.75,5,1,Active Shallow Crust,0,89.98,0
88,-5.48151,49.46184,3.75,5,1,Active Shallow Crust,0,89.98,0
"""

class TestSESCatalogue(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
        self.event_file = os.path.join(self.test_dir, 'temp_events.csv')
        self.rupture_file = os.path.join(self.test_dir, 'temp_ruptures.csv')
        self.output_file = os.path.join(self.test_dir, 'temp_output_hmtk.csv')
        
        with open(self.event_file, 'w') as f:
            f.write(EVENTS_TEXT)
            
        with open(self.rupture_file, 'w') as f:
            f.write(RUPTURES_TEXT)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_merge_ses_event_rupture(self):
        """ Testing if the merge function correctly joins events and ruptures on 'rup_id' """
        df = merge_ses_event_rupture(self.event_file, self.rupture_file)
        self.assertEqual(len(df), 10)
        self.assertIn('mag', df.columns)
        self.assertEqual(df.iloc[0]['rup_id'], 118)
        self.assertEqual(df.iloc[-1]['mag'], 3.75)

    def test_add_random_datetime_constraints(self):
        """ Testing if the random datetime generation stays within valid calendar bounds """
        df = pd.DataFrame({'year': [2020, 2021], 'event_id': [1, 2]})
        df_result = add_random_datetime(df, seed=42)
        
        self.assertTrue(df_result['month'].between(1, 12).all())
        self.assertTrue(df_result['hour'].between(0, 23).all())
        self.assertEqual(len(df_result.iloc[0]['Date']), 10)

    def test_convert_to_hmtk_mapping(self):
        """ Testing if OQ columns are correctly renamed to HMTK format """
        data = {
            'centroid_lon': [28.5], 'centroid_lat': [39.5], 'mag': [5.0], 
            'centroid_depth': [10.0], 'event_id': [1], 'year': [2020],
            'month': [1], 'day': [1], 'hour': [12], 'minute': [30], 'second': [0],
            'rup_id': [101], 'source_id': ['Src1'], 'rlz_id': [0], 'ses_id': [1],
            'multiplicity': [1], 'trt': ['Active'], 'strike': [45], 'dip': [90], 
            'rake': [0], 'Date': ['01/01/2020']
        }
        df = pd.DataFrame(data)
        hmtk_df = convert_to_hmtk(df)
        
        self.assertIn('longitude', hmtk_df.columns)
        self.assertIn('latitude', hmtk_df.columns)
        self.assertIn('magnitude', hmtk_df.columns)
        self.assertIn('eventID', hmtk_df.columns)
        
        self.assertNotIn('centroid_lon', hmtk_df.columns)

    def test_full_build_pipeline(self):
        """ Testing the complete workflow from raw ses to saved HMTK csv """
        result_path = build_hmtk_ses_catalogue(
            self.event_file, self.rupture_file, self.output_file
        )
        self.assertTrue(os.path.exists(result_path))
        
        saved_df = pd.read_csv(result_path)
        self.assertEqual(len(saved_df), 10)
        self.assertEqual(saved_df.columns[0], "longitude")

