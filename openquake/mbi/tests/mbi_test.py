
import os
import unittest
from subprocess import run

BASE = os.path.join(os.path.dirname(__file__), 'data', 'mbi')


class MBITest(unittest.TestCase):

    def test01(self):
        fname = os.path.join(BASE, 'test01.toml')
        out = run(['oqm', 'mbi', '\'{:s}\''.format(fname)],
                  check=True, capture_output=True)
        print(out)
        print(out.args)
        print('>>', out.stdout)
        msg = 'Test 1 failed'
        self.assertEqual(out.returncode, 0, msg)
