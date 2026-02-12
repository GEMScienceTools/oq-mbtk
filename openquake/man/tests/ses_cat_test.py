import os
import unittest
import pandas as pd
import numpy as np

from openquake.man.ses_cat import (
    merge_ses_event_rupture, 
    add_random_datetime, 
    convert_to_hmtk,
    build_hmtk_ses_catalogue
)

class TestSESCatalogue(unittest.TestCase):
    """ Testing the merging of rupture and event sets"""

    def setUp(self):
        """ Creating temporary rupture and event sets """
        self.event_file = 'temp_events.csv'
        self.rupture_file = 'temp_ruptures.csv'
        self.output_file = 'temp_output_hmtk.csv'
        
        events_data = ("SkipHeader\nevent_id,rup_id,year,ses_id,source_id,rlz_id\n"
                       "0,118,5948,1,fr,0\n"
                       "1,65,3266,12,fr,1\n"
                       "2,91,4567,18,fr,2\n"
                       "3,136,6815,24,fr,3\n"
                       "4,11,554,121,fr,4\n"
                       "5,195,9763,123,fr,5\n"
                       "6,142,7128,214,fr,6\n"
                       "7,177,8881,349,fr,7\n"
                       "8,150,7515,504,fr,8\n"
                       "9,88,4425,504,fr,9")
        with open(self.event_file, 'w') as f:
            f.write(events_data)
            
        rupture_data = ("SkipHeader\nrup_id,centroid_lon,centroid_lat,mag,centroid_depth,"
                        "multiplicity,trt,strike,dip,rake\n"
                        "118,4.5528,44.82738,3.55,5,1,Active Shallow Crust,0,89.99,0\n"
                        "65,4.5528,44.82738,3.65,5,1,Active Shallow Crust,0,89.98,0\n"
                        "91,4.5528,44.82738,3.65,5,1,Active Shallow Crust,0,89.98,-90\n"
                        "136,4.5528,44.82738,3.75,5,1,Active Shallow Crust,0,89.98,0\n"
                        "11,4.5528,44.82738,4.55,15,1,Active Shallow Crust,0,89.99,0\n"
                        "195,4.5528,44.82738,4.55,5,1,Active Shallow Crust,240.0,57.99,0\n"
                        "142,4.5528,44.82738,5.25,15,1,Active Shallow Crust,128.8,89.98,0\n"
                        "177,4.5528,44.82738,6.45,15,1,Active Shallow Crust,0,89.99,0\n"
                        "150,-5.48151,49.46184,3.75,5,1,Active Shallow Crust,0,89.98,0\n"
                        "88,-5.48151,49.46184,3.75,5,1,Active Shallow Crust,0,89.98,0")
        with open(self.rupture_file, 'w') as f:
            f.write(rupture_data)

    def tearDown(self):
        """ Cleaning up temporary files after tests are completed """
        files_to_remove = [self.event_file, self.rupture_file, self.output_file]
        for f in files_to_remove:
            if os.path.exists(f):
                os.remove(f)

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

if __name__ == '__main__':
    unittest.main()