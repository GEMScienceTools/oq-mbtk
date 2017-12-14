
import os
import unittest
import logging

import oqmbt.tools.notebook as nb

from oqmbt.oqt_project import OQtProject
from oqmbt.notebooks.project.project_create import project_create
from oqmbt.tests.tools import delete_and_create_project_dir


class TestLoadGeometryFromShapefile(unittest.TestCase):

    BASE_DATA_PATH = os.path.dirname(__file__)

    def setUp(self):
        #
        #
        fname = './testloadgeometryfromshapefile.log'
        logging.basicConfig(filename=fname,level=logging.DEBUG)
        #
        # clear directory
        folder = os.path.join(self.BASE_DATA_PATH, './../../tmp/project_test')
        delete_and_create_project_dir(folder)
        #
        # set environment variable
        self.prj_path = os.path.join(self.BASE_DATA_PATH,
                                     './../../tmp/project_test/test.oqmbtp')
        os.environ["OQMBT_PROJECT"] = self.prj_path
        #
        # create the project
        path = './../data/project.ini'
        inifile = os.path.join(self.BASE_DATA_PATH, path)
        project_create([inifile, os.path.dirname(self.prj_path)])
        #
        # add to the model the name of the shapefile - the path is relative to
        # the position of the project file
        oqtkp = OQtProject.load_from_file(self.prj_path)
        model_id = 'model01'
        oqtkp.active_model_id = model_id
        model = oqtkp.models[model_id]
        path = './../../notebooks/data/shapefiles/area_sources.shp'
        model.area_shapefile_filename = path
        oqtkp.models[model_id] = model
        oqtkp.save()

    def test_01(self):
        #
        # running the notebook
        nb_name = 'load_geometry_from_shapefile.ipynb'
        nb_path = './../../../notebooks/sources_area/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        nb.run(nb_full_path, '')
        #
        # loading the project
        oqtkp = OQtProject.load_from_file(self.prj_path)
        model_id = 'model01'
        oqtkp.active_model_id = model_id
        model = oqtkp.models[model_id]
        #
        # checking the number and the type of sources loaded
        self.assertEqual(len(model.sources), 2)
        keys = list(model.sources.keys())
        self.assertEqual(model.sources[keys[0]].source_type, 'AreaSource')
        self.assertEqual(model.sources[keys[1]].source_type, 'AreaSource')

        #assert 0 == 1
