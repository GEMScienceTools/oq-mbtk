import os
import unittest
import logging

import openquake.mbt.tools.notebook as nb

from openquake.mbt.tools import automator
from openquake.mbt.tools.utils import GetSourceIDs
from openquake.mbt.oqt_project import OQtProject
from openquake.mbt.notebooks.project.project_create import project_create
from openquake.mbt.tests.tools.tools import delete_and_create_project_dir


class TestWorkflowSmoothedNoFaults(unittest.TestCase):

    BASE_DATA_PATH = os.path.dirname(__file__)

    def setUp(self):
        #
        #
        fname = './wf03.log'
        # logging.basicConfig(filename=fname, level=logging.DEBUG)
        logging.basicConfig(filename=fname, level=logging.WARN)
        #
        # clear directory where the project will be created
        folder = os.path.join(self.BASE_DATA_PATH, '..', 'tmp', 'project_test')
        delete_and_create_project_dir(folder)
        #
        # set environment variable
        self.prj_path = os.path.join(folder, 'test.oqmbtp')
        os.environ["OQMBT_PROJECT"] = self.prj_path
        #
        # create the project
        inifile = os.path.join(self.BASE_DATA_PATH, '..', 'data', 'wf01',
                               'project.ini')
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
        model.area_shapefile_filename = os.path.join('.', '..', '..', 'data',
                                                     'wf01', 'shapefiles',
                                                     'test_area.shp')
        #
        # set the catalogue name
        path = './../../data/wf01/catalogue.csv'
        model.catalogue_csv_filename = path

        # required by mfd_double_truncated_from_slip_rate_SRC.ipynb
        model.default_bgr = 1.0

        # required by compute_mo_from_strain.ipynb
        model.shear_modulus = 3.2e10

        # required by compute_mo_from_strain.ipynb
        model.coup_coef = 0.8

        # required by compute_mo_from_strain.ipynb
        model.coup_thick = 15.0

        # required by compute_mo_from_strain.ipynb
        model.strain_cell_dx = 0.250
        model.strain_cell_dy = 0.200

        model.area_discretization = 10.0

        # required by set_mfd_tapered_GR.ipynb
        model.m_min = 5.0

        # required by set_mfd_tapered_GR.ipynb
        model.bin_width = 0.1

        model.faults_lower_threshold_magnitude = 6.5
        model.msr = 'WC1994'

        # required by set_mem_from_seismicity_max_obs_plus_delta.ipynb
        model.magnitude_max_delta = 0.3

        # required by create_sources_no_faults.ipynb
        model.smoothing_param = [['gaussian', 50, 20, 0.95],
                                 ['gaussian', 20,  5, 0.05]]
        model.lower_seismogenic_depth = 30.0

        #
        # create the hypo files - the folder hypo_depths is created by the
        # 'project_create' script
        folder = os.path.dirname(self.prj_path)
        for i in [1, 2, 3]:
            fname = 'hypo_depths-model01-{:d}.csv'.format(i)
            path = os.path.join(folder, 'hypo_depths', fname)
            f = open(path, 'w')
            f.write('depth,weight\n')
            f.write('10,0.6\n')
            f.write('20,0.4\n')
            f.close()
        model.hypo_dist_filename = 'model01_hypo_dist.hdf5'
        #
        # create the focal mechanism files
        for i in [1, 2, 3]:
            fname = 'focal_mechs-model01-{:d}.csv'.format(i)
            path = os.path.join(folder, 'focal_mechs', fname)
            f = open(path, 'w')
            f.write('strike,dip,rake,weight\n')
            f.write('0.00,90.00,0.00,1.00\n')
            f.close()
        model.nodal_plane_dist_filename = 'model01_focal_mech_dist.hdf5'
        #
        # saving the project
        oqtkp.models[model_id] = model
        oqtkp.save()

    def test_01(self):
        """
        This implements a workflow creating area sources without faults
        """
        reports_folder = os.path.join('..', 'tmp', 'project_test', 'reports')
        #
        #
        oqtkp = OQtProject.load_from_file(self.prj_path)
        model = oqtkp.models['model01']
        get_src_ids = GetSourceIDs(model)
        #
        # AREA SOURCES
        # .....................................................................
        # running the first notebook that loads the geometry of the sources
        # from the shapefile
        nb_name = 'load_geometry_from_shapefile.ipynb'
        nb_path = os.path.join('..', '..', 'notebooks', 'sources_area')
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        nb.run(nb_full_path, '')

        del oqtkp
        oqtkp = OQtProject.load_from_file(self.prj_path)
        model = oqtkp.models['model01']
        #
        # .....................................................................
        # set tectonic-region
        oqtkp.set_tectonic_region('model01', 'Active Shallow Crust')
        oqtkp.save()
        # checking
        del oqtkp
        oqtkp = OQtProject.load_from_file(self.prj_path)
        oqtkp.active_model_id = 'model01'
        model = oqtkp.models['model01']

        for key in sorted(model.sources):
            assert model.sources[key].tectonic_region_type == \
                'Active Shallow Crust'

        oqtkp = OQtProject.load_from_file(self.prj_path)
        oqtkp.active_model_id = 'model01'
        model = oqtkp.models['model01']
        #
        # .....................................................................
        # catalogue pre-processing
        nb_name = 'catalogue_pre_processing.ipynb'
        nb_path = os.path.join('..', '..', 'notebooks', 'catalogue')
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        nb.run(nb_full_path, '')
        #
        # .....................................................................
        # assign default completeness to all the sources
        nb_name = 'set_completeness_to_all_area_sources.ipynb'
        nb_path = os.path.join('..', '..', 'notebooks', 'sources_area')
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        nb.run(nb_full_path, '')
        #
        # .....................................................................
        # calculate GR parameters for all the area sources
        #
        # loading the project
        oqtkp = OQtProject.load_from_file(self.prj_path)
        oqtkp.active_model_id = 'model01'
        model = oqtkp.models['model01']
        #
        # running notebook
        get_src_ids = GetSourceIDs(model)
        get_src_ids.keep_equal_to('source_type', ['AreaSource'])
        nb_name = 'compute_double_truncated_GR_from_seismicity.ipynb'
        nb_path = os.path.join('..', '..', 'notebooks', 'sources_area')
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        automator.run(self.prj_path, 'model01', nb_full_path, get_src_ids.keys,
                      reports_folder=reports_folder)
        #
        # .....................................................................
        # upload hypocentral depths
        #
        # loading the project
        del oqtkp
        oqtkp = OQtProject.load_from_file(self.prj_path)
        oqtkp.active_model_id = 'model01'
        model = oqtkp.models['model01']
        #
        # running notebook
        get_src_ids = GetSourceIDs(model)
        get_src_ids.keep_equal_to('source_type', ['AreaSource'])
        nb_name = 'load_hypocentral_depth_distribution_from_csv.ipynb'
        nb_path = os.path.join('..', '..', 'notebooks', 'sources_area')
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        automator.run(self.prj_path, 'model01', nb_full_path, get_src_ids.keys)
        #
        # .....................................................................
        # upload focal mechanism distribution
        #
        # loading the project
        del oqtkp
        oqtkp = OQtProject.load_from_file(self.prj_path)
        oqtkp.active_model_id = 'model01'
        model = oqtkp.models['model01']
        #
        # running notebook
        get_src_ids = GetSourceIDs(model)
        get_src_ids.keep_equal_to('source_type', ['AreaSource'])
        nb_name = 'load_nodal_plane_distribution_from_csv.ipynb'
        nb_path = os.path.join('..', '..', 'notebooks', 'sources_area')
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        automator.run(self.prj_path, 'model01', nb_full_path, get_src_ids.keys)
        #
        # .....................................................................
        # Setting the mmax
        get_src_ids = GetSourceIDs(model)
        get_src_ids.keep_equal_to('source_type', ['AreaSource'])
        nb_name = 'set_mem_from_seismicity_max_obs_plus_delta.ipynb'
        nb_path = os.path.join('..', '..', 'notebooks', 'sources_area')
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        automator.run(self.prj_path, 'model01', nb_full_path, get_src_ids.keys)
        #
        # .....................................................................
        # Setting the MFD
        get_src_ids = GetSourceIDs(model)
        get_src_ids.keep_equal_to('source_type', ['AreaSource'])
        nb_name = 'set_mfd_double_truncated_GR.ipynb'
        nb_path = os.path.join('..', '..', 'notebooks', 'sources_area')
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        automator.run(self.prj_path, 'model01', nb_full_path, get_src_ids.keys)
        #
        # .....................................................................
        # Creating sources
        nb_name = 'create_sources_no_faults.ipynb'
        nb_path = os.path.join('..', '..', 'notebooks',
                               'sources_distributed_s')
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        automator.run(self.prj_path, 'model01', nb_full_path, get_src_ids.keys)
