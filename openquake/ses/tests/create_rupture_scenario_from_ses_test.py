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
import shutil
import pathlib
import unittest
import tempfile
import numpy as np
import pandas as pd
import subprocess

from openquake.ses.create_rupture_scenario_from_ses import process

from openquake.calculators import base
from openquake.commonlib import readinput, logs

# This file folder
TFF = pathlib.Path(__file__).parent.resolve()


class ScenarioCreationTestCase(unittest.TestCase):

    def test_scenario_simplefault(self):

        # Create a datastore with 69 ruptures
        fname_ini = TFF / 'd_test01' / 'test01.ini'
        dstore = base.dcache.get(str(fname_ini))

        # Read the table to get path to the hdf5 file
        df = pd.read_csv(base.dcache.ini_hdf5_csv, names=['ini', 'hdf5'])
        idx = np.where(df['ini'] == str(fname_ini))[0]

        # Create temporary folder
        tmpdir = pathlib.Path(tempfile.mkdtemp())

        # Copy ini file
        shutil.copyfile(TFF / 'd_scen' / 'job.ini', tmpdir / 'job.ini')

        # Set the configuration dictionary
        cfg = {'datastore': df['hdf5'][idx].values[0],
               'rupture_id': 1073741899,
               'with_between_within_ratio': 1.4,
               'output_rupture_csv': str(tmpdir / 'rupture_model.csv'),
               'output_gmclt_xml': str(tmpdir / 'gmclt.xml')
               }

        # Create the scenario rupture and GMC logic tree
        process(cfg)

        # Check the results
        kw = {}
        params = readinput.get_params(str(tmpdir / 'job.ini'), kw)
        log = logs.init(params)
        oq = log.get_oqparam()

        calc = base.calculators(oq, log.calc_id)
        calc.test_mode = True
        with calc._monitor:
            result = calc.run()
        ds_res = calc.datastore

        # Check output
        self.assertEqual(ds_res['ruptures'][0][4], np.uint8(161))
        expected = np.array(
                [0.012436, 0.024435, 0.023045, 0.009636],
                dtype=np.float32)
        np.testing.assert_array_almost_equal(ds_res['avg_gmf'][0, 0], expected)


    def test_scenario_area(self):

        # Create a datastore with 69 ruptures
        fname_ini = TFF / 'd_test02' / 'test02.ini'
        dstore = base.dcache.get(str(fname_ini))

        # Read the table to get path to the hdf5 file
        df = pd.read_csv(base.dcache.ini_hdf5_csv, names=['ini', 'hdf5'])
        idx = np.where(df['ini'] == str(fname_ini))[0]

        # Create temporary folder
        tmpdir = pathlib.Path(tempfile.mkdtemp())

        # Copy ini file
        shutil.copyfile(TFF / 'd_scen' / 'job.ini', tmpdir / 'job.ini')

        # Set the configuration dictionary
        cfg = {'datastore': df['hdf5'][idx].values[0],
               'rupture_id': 35,
               'with_between_within_ratio': 1.4,
               'output_rupture_csv': str(tmpdir / 'rupture_model.csv'),
               'output_gmclt_xml': str(tmpdir / 'gmclt.xml')
               }

        # Create the scenario rupture and GMC logic tree
        process(cfg)

        # TODO Check the results
        kw = {}
        params = readinput.get_params(str(tmpdir / 'job.ini'), kw)
        log = logs.init(params)
        oq = log.get_oqparam()

        calc = base.calculators(oq, log.calc_id)
        calc.test_mode = True
        with calc._monitor:
            result = calc.run()
        ds_res = calc.datastore

        # Check output
        self.assertEqual(ds_res['ruptures'][0][4], np.uint8(153))
        expected = np.array(
                [0.014163, 0.024379, 0.014675, 0.010099],
                dtype=np.float32)
        np.testing.assert_array_almost_equal(ds_res['avg_gmf'][0, 0], expected)
