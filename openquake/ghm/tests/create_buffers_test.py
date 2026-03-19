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

"""
Module create_buffer_test
"""

import toml
import pathlib
import tempfile
import unittest

from openquake.ghm.create_buffers import main

HERE = pathlib.Path(__file__).parent.resolve()


class BufferCreateTestCase(unittest.TestCase):
    """ Check the construction of buffers """

    def test_simple_test(self):

        regenerate = False

        # Reading original config and fix paths etc.
        fname_conf = HERE / 'data' / 'buffer' / 'config.toml'

        # NOTE: this recreates the expected file
        if regenerate:

            main(fname_conf)

        else:

            config = toml.load(fname_conf)
            config['output_dir'] = tempfile.mkdtemp()
            fname_path = pathlib.Path(fname_conf)
            ROOT = fname_path.resolve().parent
            config['zonation_fname'] = str(ROOT / config['zonation_fname'])
            _, new_config = tempfile.mkstemp(suffix='.toml')

            # Writing the new config
            with open(new_config, 'w') as f:
                _ = toml.dump(config, f)
            print(f'New config: {new_config}')
            main(new_config)

            # Testing
            computed = pathlib.Path(config['output_dir']) / 'buffer_100km.geojson'
            expected = ROOT / 'out' / 'buffer_100km.geojson'
            self.assertListEqual(list(open(computed)), list(open(expected)))
