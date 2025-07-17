import unittest
import os
import tempfile

from openquake.mbi.wkf import create_nrml_sources
from openquake.hazardlib import nrml
from openquake.hazardlib.sourceconverter import SourceConverter

BASE = os.path.join(os.path.dirname(__file__))


class MakeSourcesTestCase(unittest.TestCase):
       
   def test_nrml_src_scalerel(self):
       config = os.path.join(BASE, 'data', 'mbi', 'config_nrml.toml')
       pattern = os.path.join(BASE, 'data', 'mbi', 'pt_srcs_ex1.csv')
       folder_oq = tempfile.mkdtemp()
       nrml_srcs = create_nrml_sources.create_nrml_sources(pattern, config, folder_oq, True)
       
       # load src
       conv = SourceConverter(50., 1., 20, 0.1, 10.)
       src = nrml.to_python(os.path.join(folder_oq, 'src_pt_srcs_ex1.xml'))
       for srcs in src.src_groups:
           for src_z in srcs:
               self.assertEqual(str(src_z.magnitude_scaling_relationship), 'Leonard2014_Interplate')

