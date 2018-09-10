import unittest
import os
import json
import filecmp

import openquake.mbt.tools.fault_modeler.fault_source_modeler as fsm

# -----------------------------------------------------------------------------

BASE_DATA_PATH = os.path.dirname(__file__)

# -----------------------------------------------------------------------------

class TestDatabaseIO(unittest.TestCase):

    geojson_file = os.path.join(BASE_DATA_PATH, 'Data',
                                'ne_asia_faults_rates.geojson')

    param_map = {'source_id': 'ogc_fid',
                 'name': 'ns_name',
                 'average_dip': 'ns_average_dip',
                 'average_rake': 'ns_average_rake',
                 'net_slip_rate': 'ns_net_slip_rate',
                 'vert_slip_rate': 'ns_vert_slip_rate',
                 'strike_slip_rate': 'ns_strike_slip_rate',
                 'shortening_rate': 'ns_shortening_rate',
                 'dip_dir': 'ns_dip_dir',
                 'dip_slip_rate': 'ns_dip_slip_rate'}

    def test_fault_database(self):

        # Target and reference files
        test_file = os.path.join(BASE_DATA_PATH, 'Data',
                                 'fault_database.test.geojson')
        base_file = os.path.join(BASE_DATA_PATH, 'Data',
                                 'fault_database.base.geojson')

        # Import the database
        fault_db = fsm.FaultDatabase()
        fault_db.import_from_geojson(self.geojson_file,
                                     param_map=self.param_map,
                                     select_list=[1, 2])

        # Adding a key/value to all faults
        fault_db.add_property('lower_seismogenic_depth', value=25)

        # Adding key/value to given faults
        fault_db.add_property('M_max', value=7., id=1)
        fault_db.add_property('M_max', value=7.5, id=2)

        fault_db.remove_property('name')

        # Export the augmented database
        fault_db.export_to_geojson(test_file)

        with open(test_file, 'r') as f:
            test_out = json.load(f)

        with open(base_file, 'r') as f:
            base_out = json.load(f)

        for bo, to in zip(base_out['features'], test_out['features']):
            self.assertTrue(bo == to)

    def test_build_model_from_db(self):

        # Target and reference files
        test_file = os.path.join(BASE_DATA_PATH, 'Data',
                                 'fault_model_01.test.xml')
        base_file = os.path.join(BASE_DATA_PATH, 'Data', 
                                 'fault_model_01.base.xml')

        # Import the database
        fault_db = fsm.FaultDatabase()
        fault_db.import_from_geojson(self.geojson_file,
                                     param_map=self.param_map,
                                     select_list=[1, 2])

        # Create and export the model
        fsm.build_model_from_db(fault_db, xml_output=test_file)

        # Compare files
        self.assertTrue(filecmp.cmp(base_file, test_file))

    def test_build_source_model_single_args(self):

        # Target and reference files
        test_file = os.path.join(BASE_DATA_PATH, 'Data',
                                 'fault_model_02.test.xml')
        base_file = os.path.join(BASE_DATA_PATH, 'Data', 
                                 'fault_model_02.base.xml')

        # Create and export the model
        fsm.build_fault_model(geojson_file=self.geojson_file,
                              xml_output=test_file,
                              black_list=[1,2,3],
                              param_map=self.param_map,
                              M_max=8.2,
                              lower_seismogenic_depth=30.)

        # Compare files
        self.assertTrue(filecmp.cmp(base_file, test_file))

    def test_build_source_model_dictionary(self):

        # Target and reference files
        test_file = os.path.join(BASE_DATA_PATH, 'Data',
                                 'fault_model_03.test.xml')
        base_file = os.path.join(BASE_DATA_PATH, 'Data', 
                                 'fault_model_03.base.xml')

        # Create and export the model
        fsm.build_fault_model(geojson_file=self.geojson_file,
                              xml_output=test_file,
                              param_map=self.param_map,
                              defaults={'upper_seismogenic_depth':10.,
                                        'lower_seismogenic_depth':30.})

        # Compare files
        self.assertTrue(filecmp.cmp(base_file, test_file))

    def test_build_source_model_config_file(self):

        # Configuration, target and reference files
        conf_file = os.path.join(BASE_DATA_PATH, 'Data', 'config.ini')
        test_file = os.path.join(BASE_DATA_PATH, 'Data',
                                 'fault_model_04.test.xml')
        base_file = os.path.join(BASE_DATA_PATH, 'Data', 
                                 'fault_model_04.base.xml')

        # Create and export the model
        fsm.build_fault_model(cfg_file=conf_file)

        # Compare files
        self.assertTrue(filecmp.cmp(base_file, test_file))

if __name__ == "__main__":
    unittest.main()
