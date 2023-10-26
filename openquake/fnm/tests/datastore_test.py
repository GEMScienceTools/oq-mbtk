# ------------------- The OpenQuake Model Building Toolkit --------------------
# ------------------- FERMI: Fault nEtwoRks ModellIng -------------------------
# Copyright (C) 2023 GEM Foundation
#         .-.
#        /    \                                        .-.
#        | .`. ;    .--.    ___ .-.     ___ .-. .-.   ( __)
#        | |(___)  /    \  (   )   \   (   )   '   \  (''")
#        | |_     |  .-. ;  | ' .-. ;   |  .-.  .-. ;  | |
#       (   __)   |  | | |  |  / (___)  | |  | |  | |  | |
#        | |      |  |/  |  | |         | |  | |  | |  | |
#        | |      |  ' _.'  | |         | |  | |  | |  | |
#        | |      |  .'.-.  | |         | |  | |  | |  | |
#        | |      '  `-' /  | |         | |  | |  | |  | |
#       (___)      `.__.'  (___)       (___)(___)(___)(___)
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

import toml
import pathlib
import unittest
import tempfile
import numpy as np
import openquake.fnm.datastore as ds

from openquake.fnm.fault_system import get_rups_fsys
from openquake.fnm.importer import kite_surfaces_from_geojson
from openquake.fnm.datastore import KEYS

HERE = pathlib.Path(__file__).parent


def _get_data(settings, root):
    # Get the name of the file with faults.
    fname_faults = root / settings["general"]["fname_sections"]

    # Create surfaces using the information in the .geojson file
    surfs = kite_surfaces_from_geojson(fname_faults)

    # Create ruptures
    return get_rups_fsys(surfs, settings)


def _get_settings():
    tmp = pathlib.Path(tempfile.mkdtemp())
    fname = tmp / "test.hdf5"

    # Removing the file is it exists
    if pathlib.Path(fname).exists():
        pathlib.Path.unlink(fname)

    # Read the configuration
    settings_fname = HERE / "data" / "config_export.toml"
    settings = toml.load(settings_fname)

    # Path to the settings file. This is the root folder.
    root = pathlib.Path(settings_fname).parent

    return settings, fname, root


@unittest.skip("Slooowww")
class TestSaveDatastore(unittest.TestCase):
    def test_write_read_datastore(self):
        # Get the settings
        settings, fname, root = _get_settings()

        # Getting data to be stored in the datastore
        out = _get_data(settings, root)

        # Write output
        ds.write(fname, out)

        # Read output
        stored = ds.read(fname)

        # Checking that what is stored corresponds to what we provided to the
        # function that writes the datastore
        for key in KEYS:
            if isinstance(KEYS[key], str):
                expected = out[key]
                computed = stored[key]
                if isinstance(expected, np.ndarray):
                    np.testing.assert_array_almost_equal(computed, expected)
