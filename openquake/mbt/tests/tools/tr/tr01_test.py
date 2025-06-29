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
import numpy
import configparser
import pandas as pd
import shutil
import h5py

from openquake.mbi.ccl.classify import classify

BASE_PATH = os.path.dirname(__file__)

PLOT = False
try:
    import glob
    import pyvista as pv
    from openquake.plt.faults import get_pv_points, get_pv_line
except ImportError:
    PLOT = False


class TrTestCase(unittest.TestCase):
    """
    This class tests the tectonic regionalisation workflow.
    """

    def setUp(self):

        self.root_folder = os.path.join(BASE_PATH)
        self.ini_fname = os.path.join(
            BASE_PATH, '..', '..', 'data', 'tr01', 'tr01.ini')
        self.treg_filename = os.path.join(
            BASE_PATH, '..', '..', 'tmp', 'test02.hdf5')

        self.tmp = os.path.join(BASE_PATH, '..', '..', 'tmp')
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)
        os.makedirs(self.tmp)

        self.config = configparser.ConfigParser()
        self.config.read(self.ini_fname)

    def tearDown(self):
        # Remove tmp folder
        shutil.rmtree(self.tmp)

    def testcase01(self):
        """
        Testing TR
        """

        PLOT = False
        if PLOT:
            _plot(self.ini_fname, self.config)

        # Run classification
        classify(self.ini_fname, True, self.root_folder)
        f = h5py.File(self.treg_filename, 'r')

        # Testing Crustal Seismicity
        msg = 'Indexes of different elements: \n'
        expected = [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
        expected = [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]

        computed = f['crustal'][:].astype(int)
        dff = numpy.where(numpy.abs(expected - computed) > 0)
        if len(dff[0]):
            for val in dff[0]:
                msg += f'{val:.0f}'
        numpy.testing.assert_array_equal(computed, expected, err_msg=msg)

        # testing interface
        msg = 'Indexes of different elements: \n'
        expected = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0]
        computed = f['int_cam'][:].astype(int)
        if len(dff[0]):
            for val in dff[0]:
                msg += f'{val:.0f}'
        numpy.testing.assert_array_equal(computed, expected, err_msg=msg)

        # testing slab
        expected = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
        computed = f['slab_cam'][:].astype(int)
        numpy.testing.assert_array_equal(computed, expected)
        f.close()


def _plot(ini_fname, config):
    scl = -0.01
    root = os.path.dirname(ini_fname)

    tmp = config['general']['catalogue_filename']
    fname = os.path.join(root, tmp)
    df = pd.read_csv(fname, delimiter=',')
    cat = numpy.zeros((len(df), 3))
    cat[:, 0] = df.longitude.to_numpy()
    cat[:, 1] = df.latitude.to_numpy()
    cat[:, 2] = df.depth.to_numpy() * scl
    size = numpy.ones((len(df))) * 0.1

    # Plot hypocenters
    pl = pv.Plotter()
    pdata = get_pv_points(cat, size)
    pl.add_mesh(pdata, color='red')

    pdata = pv.PolyData(cat)
    pdata["labels"] = [f"{i}" for i in range(len(cat))]
    pl.add_point_labels(pdata, "labels", font_size=36)

    # Plot profiles interface
    tmp_path = config['int_cam']['folder']
    pattern = os.path.join(root, tmp_path, 'edge_*.csv')
    for fname in glob.glob(pattern):
        dat = numpy.loadtxt(fname)
        dat[:, 2] *= scl
        pdata = get_pv_line(dat)
        pl.add_mesh(pdata, color='green')

    # Plot profiles interface - Below
    tmp_path = config['int_cam']['folder']
    pattern = os.path.join(root, tmp_path, 'edge_*.csv')
    for fname in glob.glob(pattern):
        dat = numpy.loadtxt(fname)
        val = float(config['int_cam']['distance_buffer_below'])
        dat[:, 2] += val
        dat[:, 2] *= scl
        pdata = get_pv_line(dat)
        pl.add_mesh(pdata, color='lightgreen')

    # Plot profiles slab
    tmp_path = config['slab_cam']['folder']
    pattern = os.path.join(root, tmp_path, 'edge_*.csv')
    for fname in glob.glob(pattern):
        dat = numpy.loadtxt(fname)
        dat[:, 2] *= scl
        pdata = get_pv_line(dat)
        pl.add_mesh(pdata, color='blue')

    pl.view_isometric()
    pl.set_viewup((0, 0, 1))
    pl.show_grid()
    pl.show(interactive=True)
