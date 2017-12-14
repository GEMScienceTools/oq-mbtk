
import os
import unittest
import logging

import oqmbt.tools.notebook as nb

from oqmbt.tools import automator
from oqmbt.tools.utils import GetSourceIDs
from oqmbt.oqt_project import OQtProject
from oqmbt.notebooks.project.project_create import project_create
from oqmbt.tests.tools import delete_and_create_project_dir

from oqmbt.tools.completeness import set_completeness_for_sources

TABLE = [[1985., 5.0],
         [1970., 5.3],
         [1923., 5.6],
         [1900., 6.6],
         [1800., 7.9]]


class TestComputeDoubleTruncatedGR(unittest.TestCase):

    BASE_DATA_PATH = os.path.dirname(__file__)

    def setUp(self):
        #
        #
        fname = './testcomputedoubletruncatedGRfromseismicity.log'
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
        #
        # set the shapefile with the geometry of area sources
        path = './../../notebooks/data/shapefiles/area_sources.shp'
        model.area_shapefile_filename = path
        #
        # set the catalogue name
        path = './../../notebooks/data/catalogue.csv'
        model.catalogue_csv_filename = path
        #
        # saving the project
        oqtkp.models[model_id] = model
        oqtkp.save()

    def test_01(self):
        """
        Read geometry from a shapefile and compute a MFD for each source
        """
        #
        #
        oqtkp = OQtProject.load_from_file(self.prj_path)
        model = oqtkp.models['model01']
        get_src_ids = GetSourceIDs(model)
        #
        # running the first notebook that loads the geometry of the sources
        # from the shapefile
        nb_name = 'load_geometry_from_shapefile.ipynb'
        nb_path = './../../../notebooks/sources_area/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        nb.run(nb_full_path, '')
        #
        # create the pickled catalogue
        nb_name = 'catalogue_create_pickle.ipynb' # we miss the completeness t
        nb_path = './../../../notebooks/catalogue/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        nb.run(nb_full_path, '')
        #
        # set the completeness table for the to sources
        set_completeness_for_sources(TABLE, ['02', '03'])
        #
        # checking the creation of the pickled version of the catalogue
        file_name = 'model01_catalogue.pkl'
        file_path = './../../tmp/project_test/'
        tmp = os.path.join(self.BASE_DATA_PATH, file_path, file_name)
        nb_full_path = os.path.abspath(tmp)
        assert os.path.exists(nb_full_path)
        #
        # loading the project
        oqtkp = OQtProject.load_from_file(self.prj_path)
        oqtkp.active_model_id = 'model01'
        model = oqtkp.models['model01']
        #
        # running the notebook that computes the GR parameters
        get_src_ids = GetSourceIDs(model)
        get_src_ids.keep_equal_to('source_type', ['AreaSource'])
        nb_name = 'compute_double_truncated_GR_from_seismicity.ipynb'
        nb_path = './../../../notebooks/sources_area/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        reports_folder = './../../tmp/project_test/reports/'
        automator.run(self.prj_path, 'model01', nb_full_path, get_src_ids.keys,
                      reports_folder=reports_folder)
        #
        # loading the project
        oqtkp = OQtProject.load_from_file(self.prj_path)
        oqtkp.active_model_id = 'model01'
        model = oqtkp.models['model01']
        #
        # check the a and b values computed
        keys = list(model.sources.keys())
        print (model.sources['02'].__dict__)
        print (model.sources['03'].__dict__)
        self.assertAlmostEqual(model.sources['02'].a_gr, 5.824285578226533)
        self.assertAlmostEqual(model.sources['02'].b_gr, 1.1421442004454874)
        self.assertAlmostEqual(model.sources['03'].a_gr, 3.8036622760645868)
        self.assertAlmostEqual(model.sources['03'].b_gr, 0.8832991033938602)
