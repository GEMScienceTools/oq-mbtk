# coding: utf-8

import pathlib
import tempfile
import unittest
import numpy as np
from openquake.fnm.interface.create_ruptures import main as create_ruptures

HERE = pathlib.Path(__file__).parent


class TestCreateRuptures(unittest.TestCase):

    def test_create_ruptures(self):
        """Test the CLI interface for the creation of ruptures"""

        fname_config = HERE / 'data' / 'config01.toml'

        # Set a temporary output
        tmp = pathlib.Path(tempfile.mkdtemp())
        fname = tmp / 'test.hdf5'

        # Removing the file is it exists
        if pathlib.Path(fname).exists():
            pathlib.Path.unlink(fname)

        # Create the ruptures
        tmp = {'output_datastore': str(fname)}
        create_ruptures(fname_config, general=tmp)
