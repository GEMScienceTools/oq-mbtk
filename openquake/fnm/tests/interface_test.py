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

import pathlib
import tempfile
import unittest
from openquake.fnm.interface.create_and_store_ruptures import (
    main as create_ruptures,
)

HERE = pathlib.Path(__file__).parent


@unittest.skip("deprecated")
class TestCreateRuptures(unittest.TestCase):
    def test_create_ruptures(self):
        """Test the CLI interface for the creation of ruptures"""

        fname_config = HERE / "data" / "config01.toml"

        # Set a temporary output
        tmp = pathlib.Path(tempfile.mkdtemp())
        fname = tmp / "test.hdf5"

        # Removing the file is it exists
        if pathlib.Path(fname).exists():
            pathlib.Path.unlink(fname)

        # Create the ruptures
        tmp = {"output_datastore": str(fname)}
        create_ruptures(fname_config, general=tmp)
