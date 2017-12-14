
import os
import unittest

from oqmbt.notebooks.project.project_create import project_create


class TestCreateProject(unittest.TestCase):

    BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), '../data')

    def test_create_01(self):
        """
        Create an oqmbt project
        """
        inifile = filename = os.path.join(self.BASE_DATA_PATH, 'project.ini')
        argv = [inifile]
        project_create(argv)
        #
        # files created with the construction of the project
        fles = ['eqk_rates.hdf5', 'hypo_close_to_faults.hdf5', 'test.oqmbtp',
                'hypo_depths.hdf5', 'completeness.hdf5']
        #
        # checking files existence
        folder = './../../tmp/project_test'
        for fle in fles:
            tmp = os.path.join(folder, fle)
            print (tmp)
            assert os.path.isfile(tmp)
        #
        # check that in the new folder there are no other files
        self.assertEqual(len(os.walk(folder).__next__()[2]), 5)
