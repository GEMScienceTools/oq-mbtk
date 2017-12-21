
import os
import re
import unittest
import logging

import oqmbt.tools.notebook as nb

from oqmbt.oqt_project import OQtProject
from oqmbt.notebooks.project.project_create import project_create
from oqmbt.tests.tools.tools import delete_and_create_project_dir


BASE_DATA_PATH = os.path.dirname(__file__)

class TestLoadSimpleFaultOQ(unittest.TestCase):

    def setUp(self):
        #
        # set logging settings
        fname = './testcomputedoubletruncatedmfd.log'
        logging.basicConfig(filename=fname,level=logging.DEBUG)
        #
        # clear directory
        folder = os.path.join(BASE_DATA_PATH, './../../tmp/project_test')
        delete_and_create_project_dir(folder)
        #
        # set environment variable
        self.prj_path = os.path.join(BASE_DATA_PATH,
                                     './../../tmp/project_test/test.oqmbtp')
        os.environ["OQMBT_PROJECT"] = self.prj_path
        #
        # create the project
        path = './../data/project.ini'
        inifile = os.path.join(BASE_DATA_PATH, path)
        project_create([inifile, os.path.dirname(self.prj_path)])
        #
        # add to the model the name of the shapefile - the path is relative to
        # the position of the project file
        oqtkp = OQtProject.load_from_file(self.prj_path)
        model_id = 'model01'
        oqtkp.active_model_id = model_id
        model = oqtkp.models[model_id]
        path = './../../data/wf01/shapefiles/test_faults.shp'
        model.faults_shp_filename = path
        oqtkp.models[model_id] = model
        oqtkp.save()


    def test_load_fault_01(self):
        """
        Load fault data from a shapefile (only simple faults for the time being)
        """
        #
        # loading the 
        nb_name = 'load_data_from_shapefile_fmg.ipynb'
        nb_path = './../../../notebooks/sources_shallow_fault/'
        tmp = os.path.join(BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        nb.run(nb_full_path, '')
        #
        # loading the project
        oqtkp = OQtProject.load_from_file(self.prj_path)
        oqtkp.active_model_id = 'model01'
        model = oqtkp.models['model01']
        #
        # count the number of area sources
        cnt = 0
        for key in list(model.sources.keys()):
            src = model.sources[key]
            if re.match('SimpleFaultSource', src.source_type):
                cnt += 1
        # checking the number of sources
        assert cnt == 6
        # checking slip rate values
        self.assertAlmostEqual(0.01, model.sources['sf395'].slip_rate, 0.01)
        self.assertAlmostEqual(0.06, model.sources['sf400'].slip_rate, 0.01)
