
import os
import h5py
import unittest
import logging
import numpy as np

import openquake.mbt.tools.notebook as nb

from openquake.mbt.tools import automator
from openquake.mbt.tools.utils import GetSourceIDs
from openquake.mbt.oqt_project import OQtProject
from openquake.mbt.notebooks.project.project_create import project_create
from openquake.mbt.tests.tools.tools import delete_and_create_project_dir

from openquake.mbt.tools.completeness import set_completeness_for_sources


class TestWorkflow(unittest.TestCase):
    """
    Test the workflow where we read information from a geojson file 
    and we build a fault source. In this case we want to set for the
    MFD of the fault the same b-value of the encompassing area source
    """

    BASE_DATA_PATH = os.path.dirname(__file__)

    def setUp(self):
        #
        #
        fname = './wf02.log'
        #logging.basicConfig(filename=fname,level=logging.DEBUG)
        logging.basicConfig(filename=fname,level=logging.WARN)
        #
        # clear directory where the project will be created
        folder = os.path.join(self.BASE_DATA_PATH, './../tmp/project_test')
        delete_and_create_project_dir(folder)
        #
        # set environment variable
        self.prj_path = os.path.join(self.BASE_DATA_PATH,
                                     './../tmp/project_test/test.oqmbtp')
        os.environ["OQMBT_PROJECT"] = self.prj_path
        #
        # create the project
        path = './../data/wf02/project.ini'
        inifile = os.path.join(self.BASE_DATA_PATH, path)
        project_create([inifile, os.path.dirname(self.prj_path)])
        #
        # load the project just created
        oqtkp = OQtProject.load_from_file(self.prj_path)
        model_id = 'model01'
        oqtkp.active_model_id = model_id
        model = oqtkp.models[model_id]
        #
        # set the shapefile with the geometry of area sources [relative path 
        # with origin the project folder]
        path = './../../data/wf02/fault.geojson'
        model.fault_geojson_filename = path
