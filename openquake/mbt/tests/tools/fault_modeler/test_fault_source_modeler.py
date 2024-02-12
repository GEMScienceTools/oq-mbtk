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
import json
import filecmp
import unittest
import configparser
import tempfile
import pathlib

import openquake.mbt.tools.fault_modeler.fault_source_modeler as fsm

# -----------------------------------------------------------------------------

BASE_DATA_PATH = os.path.dirname(__file__)

# -----------------------------------------------------------------------------


class TestDatabaseIO(unittest.TestCase):

    geojson_file = os.path.join(BASE_DATA_PATH, 'data',
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

    defaults = {'m_min': 6.0}

    def test_fault_database(self):

        # Target and reference files
        _, test_file = tempfile.mkstemp()
        base_file = os.path.join(BASE_DATA_PATH, 'data',
                                 'fault_database.base.geojson')

        # Import the database
        fault_db = fsm.FaultDatabase()
        fault_db.import_from_geojson(self.geojson_file,
                                     param_map=self.param_map,
                                     select_list=[1, 2],
                                     update_keys=True)
        # Adding a key/value to all faults
        fault_db.add_property('lower_seismogenic_depth', value=25)

        # Adding key/value to given faults
#        raise unittest.SkipTest('Marco Pagani: this test is broken!')
        fault_db.add_property('m_max', value=7., id=1)
        fault_db.add_property('m_max', value=7.5, id=2)

        fault_db.remove_property('name')

        # Export the augmented database
        fault_db.export_to_geojson(test_file)

        with open(test_file, 'r') as f:
            test_out = json.load(f)

        with open(base_file, 'r') as f:
            base_out = json.load(f)

        for bo, to in zip(base_out['features'], test_out['features']):
            self.assertTrue(bo == to)

#    @unittest.skip('find better way to compare outputs!')
    def test_build_model_from_db(self):

        # Target and reference files
        _, test_file = tempfile.mkstemp()
        base_file = os.path.join(BASE_DATA_PATH, 'data',
                                 'fault_model_01.base.xml')

        # Import the database
        fault_db = fsm.FaultDatabase()
        fault_db.import_from_geojson(self.geojson_file,
                                     param_map=self.param_map,
                                     select_list=[1, 2])

        # Create and export the model
        # this test didn't work because the length_scaling method
        # is used to get the lower_seismo_depth, and that doesn't permit
        # using the default value (the one being tested against in the base
        # output file -> 35 km). length_scaling was being set as the default
        # in build_model_from_db (fault_source_modeler line 91). I changed it
        # to 'seismo_depth' here but we should discuss the optimal default.
        # the mfds did not match because the reference file was created before
        # the implementation of m_cli, so the mfds were being computed with
        # different ranges of magnitudes, so I added the default m_min (I
        # double checked this in a separate notebook that I can share). Lastly,
        # the lons and lats and mfds didn't match because of decimal places,
        # so I changed the test to read in teh sources and compare attributes
        fsm.build_model_from_db(fault_db, xml_output=test_file,
                                param_map=self.param_map,
                                project_name='fault_model_01.test',
                                width_method='seismo_depth',
                                defaults=self.defaults)

        # Compare files
        self.assertTrue(filecmp.cmp(base_file, test_file))

    def test_build_source_model_single_args(self):

        # Target and reference files
        _, test_file = tempfile.mkstemp()
        base_file = os.path.join(BASE_DATA_PATH, 'data',
                                 'fault_model_02.base.xml')

        # Create and export the model
        # same applies here for m_min and needing to fix the lon/lat coords
        # and mfd precision
        fsm.build_fault_model(geojson_file=self.geojson_file,
                              xml_output=test_file,
                              black_list=[1, 2, 3],
                              param_map=self.param_map,
                              m_max=8.2, m_min=6.0,
                              project_name='fault_model_02.test',
                              lower_seismogenic_depth=30.)

        # Compare files
        self.assertTrue(filecmp.cmp(base_file, test_file))
        os.remove(test_file)

    def test_build_source_model_dictionary(self):

        # Target and reference files
        _, test_file = tempfile.mkstemp()
        base_file = os.path.join(BASE_DATA_PATH, 'data',
                                 'fault_model_03.base.xml')

        # Create and export the model
        # here as well with m_min and fault trace
        fsm.build_fault_model(geojson_file=self.geojson_file,
                              xml_output=test_file,
                              param_map=self.param_map,
                              project_name='fault_model_03.test',
                              defaults={'upper_seismogenic_depth': 10.,
                                        'lower_seismogenic_depth': 30.,
                                        'm_min': 6.0})

        # Compare files
        self.assertTrue(filecmp.cmp(base_file, test_file))
        os.remove(test_file)

    def test_build_source_model_config_file(self):

        # Configuration, target and reference files
        conf_file = os.path.join(BASE_DATA_PATH, 'config.ini')

        config = configparser.ConfigParser()
        config.read(conf_file)
        test_dir = pathlib.Path(tempfile.mkdtemp())
        test_file = test_dir / 'fault_model_04.test.xml'
        config['config']['xml_output'] = str(test_file)
        new_config_fname = test_dir / 'config.ini'
        data_path = pathlib.Path(BASE_DATA_PATH)
        geojson_original_path = data_path / config['config']['geojson_file']
        tmp = os.path.relpath(
            str(geojson_original_path), str(new_config_fname))
        config['config']['geojson_file'] = tmp

        with open(new_config_fname, 'w') as configfile:
            config.write(configfile)

        base_file = os.path.join(BASE_DATA_PATH, 'data',
                                 'fault_model_04.base.xml')

        # Create and export the model
        fsm.build_fault_model(cfg_file=new_config_fname)

        # Compare files
        self.assertTrue(filecmp.cmp(base_file, test_file))
