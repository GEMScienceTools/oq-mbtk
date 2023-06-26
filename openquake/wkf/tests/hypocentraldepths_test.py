#!/usr/bin/env python
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
import numpy
import tempfile
import unittest
import subprocess
import pandas as pd
import toml
import shutil

HERE = os.path.dirname(__file__)
CWD = os.getcwd()
DATA = os.path.join(os.path.dirname(__file__), 'data')

class Test_hypocentral_depth:

		def setUp(self):
			#self.out_folder = tempfile.mkdtemp()
			self.out_folder = os.path.join(DATA, "out")
			source = os.path.join(DATA, "config", "wkf_testconfig_GK.toml")
			destination = os.path.join(DATA, "config", "depth_test.toml")
			shutil.copy(source, destination)

		def test_depth_histo(self):
			source = os.path.join(DATA, "config", "wkf_testconfig_GK.toml")
			destination = os.path.join(DATA, "config", "depth_test.toml")
			shutil.copy(source, destination)
			
			config = os.path.join(DATA, "config", "depth_test.toml")
			subcatalogues_folder = os.path.join(DATA, "subcatalogues")

			depth_bins = "0.0,10.0,20.0,35.0"
			folder_figs = os.path.join(DATA, "out")
			cmd = f"oqm wkf analysis_hypocentral_depth {subcatalogues_folder} --f {folder_figs}"
			cmd = f"{cmd} --depth-bins \"{depth_bins}\" -c {config}"

			out = subprocess.run(cmd, shell=True)

			model = toml.load(config)
			obs = model['sources']['1']['hypocenter_distribution']
			exp = [ [ "0.11", "5.0",], [ "0.44", "15.0",], [ "0.45", "27.5",],]
			assert(obs == exp)
