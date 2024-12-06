# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2024 GEM Foundation
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

"""
"""

import os
import unittest
import pathlib
import tempfile

from openquake.sub.slab.rupture import calculate_ruptures
from openquake.sub.create_inslab_nrml import create
from openquake.sub.build_complex_surface import build_complex_surface
from openquake.sub.create_2pt5_model import read_profiles_csv

from openquake.hazardlib.geo.surface.kite_fault import _dbg_plot

BASE_DATA_PATH = os.path.dirname(__file__)
HERE = pathlib.Path(__file__).parent
PLOTTING = False


class RuptureCreationCAMTest(unittest.TestCase):
    """
    This set of tests check the construction of ruptures for inslab sources
    using the Central America subduction zone.
    """

    def setUp(self):

        ini_fname = pathlib.Path('./data/cam/test.ini')
        self.ini_fname = HERE / ini_fname

        # Prepare the input folder and the output folder
        in_path = os.path.join(BASE_DATA_PATH, './data/cam/sp_cam/')
        self.out_path = pathlib.Path(tempfile.mkdtemp())

        # Create the complex surface. We use the profiles used for
        # the subduction in CCARA
        max_sampl_dist = 10.
        build_complex_surface(in_path, max_sampl_dist, self.out_path,
                              upper_depth=50, lower_depth=200)
        print(self.out_path)

        if PLOTTING:
            # Read the profiles
            sps, _, _ = read_profiles_csv(in_path)
            pro = [sps[k] for k in sps.keys()]
            _dbg_plot(profs=pro, ref_idx=0)

    def step01(self):

        out_path = pathlib.Path(tempfile.mkdtemp())
        out_hdf5_fname = out_path / 'ruptures.hdf5'
        out_hdf5_smoothing_fname = out_path / 'smoothing.hdf5'
        kwargs = {'only_plt': False,
                  'profile_folder': self.out_path,
                  'out_hdf5_fname': out_hdf5_fname,
                  'out_hdf5_smoothing_fname': out_hdf5_smoothing_fname}
        calculate_ruptures(self.ini_fname, **kwargs)

        # check the existence of the rupture file
        tmps = '../data/tmp/ruptures.hdf5'
        rupture_hdf5_fname = os.path.abspath(os.path.join(BASE_DATA_PATH,
                                                          tmps))
        self.assertTrue(os.path.exists(rupture_hdf5_fname))

    def step02(self):
        label = 'test'
        tmps = '../data/tmp/ruptures.hdf5'
        rupture_hdf5_fname = os.path.abspath(os.path.join(BASE_DATA_PATH,
                                                          tmps))
        output_folder = os.path.join(BASE_DATA_PATH, '../tmp/')

        # Create nrml
        investigation_t = '1.'
        create(label, rupture_hdf5_fname, output_folder, investigation_t)

    def _steps(self):
        for name in sorted(dir(self)):
            if name.startswith("step"):
                yield name, getattr(self, name)

    def test_steps(self):
        for name, step in self._steps():
            try:
                step()
            except Exception as e:
                self.fail("{} failed ({}: {})".format(step, type(e), e))
