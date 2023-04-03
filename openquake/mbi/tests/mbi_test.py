
import os
import unittest
from subprocess import run

BASE = os.path.join(os.path.dirname(__file__), 'data', 'mbi')


class MBITest(unittest.TestCase):

    def test01(self):
        fname = os.path.join(BASE, 'test01.toml')
        out = run(['oqm', 'mbi', '{:s}'.format(fname)],
                  check=True, capture_output=True)
        msg = f'Test 1 failed. Exit status: {out}'
        self.assertEqual(out.returncode, 0, msg)
