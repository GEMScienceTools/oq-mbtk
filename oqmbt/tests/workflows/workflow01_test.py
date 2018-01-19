
import os
import h5py
import unittest
import logging
import numpy as np

import oqmbt.tools.notebook as nb

from oqmbt.tools import automator
from oqmbt.tools.utils import GetSourceIDs
from oqmbt.oqt_project import OQtProject
from oqmbt.notebooks.project.project_create import project_create
from oqmbt.tests.tools.tools import delete_and_create_project_dir

from oqmbt.tools.completeness import set_completeness_for_sources


class TestFMGWorkflow(unittest.TestCase):

    BASE_DATA_PATH = os.path.dirname(__file__)

    def setUp(self):
        #
        #
        fname = './wf01.log'
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
        path = './../data/wf01/project.ini'
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
        path = './../../data/wf01/shapefiles/test_area.shp'
        model.area_shapefile_filename = path
        #
        # set the shapefile with the geometry of fault sources [relative path 
        # with origin the project folder]
        path = './../../data/wf01/shapefiles/test_faults.shp'
        model.faults_shp_filename = path
        #
        # set the shapefile withe the faults
        path = './../../data/wf01/shapefile/test_faults.csv'
        model.catalogue_csv_filename = path
        #
        # set the catalogue name
        path = './../../data/wf01/catalogue.csv'
        model.catalogue_csv_filename = path

        model.default_bgr = 1.0 # required by imfd_double_truncated_from_slip_rate_SRC.ipynb
        model.strain_pickle_spatial_index_filename = './../../data/wf01/strain/sample_average_strain'
        model.strain_rate_model_hdf5_filename = './../../data/wf01/strain/sample_average_strain.hdf5'
        model.shear_modulus = 3.2e10 # required by compute_mo_from_strain.ipynb
        model.coup_coef = 0.8 # required by compute_mo_from_strain.ipynb
        model.coup_thick = 15.0 # required by compute_mo_from_strain.ipynb
        model.strain_cell_dx = 0.250 # required by compute_mo_from_strain.ipynb
        model.strain_cell_dy = 0.200
        model.m_min = 5.0 # required by set_mfd_tapered_GR.ipynb
        model.bin_width = 0.1 # required by set_mfd_tapered_GR.ipynb
        model.faults_lower_threshold_magnitude = 6.5 
        model.msr = 'WC1994'
        #
        # create the hypo files - the folder hypo_depths is created by the 
        # 'project_create' script
        folder = os.path.dirname(self.prj_path)
        for i in [1,2,3]:
            fname = 'hypo_depths-model01-{:d}.csv'.format(i)
            path = os.path.join(folder, 'hypo_depths', fname)
            f = open(path, 'w')
            f.write('depth,weight\n')
            f.write('10,0.6\n')
            f.write('20,0.4\n')
            f.close()
        model.hypo_dist_filename='model01_hypo_dist.hdf5'
        #
        # create the focal mechanism files
        for i in [1,2,3]:
            fname = 'focal_mechs-model01-{:d}.csv'.format(i)
            path = os.path.join(folder, 'focal_mechs', fname)
            f = open(path, 'w')
            f.write('strike,dip,rake,weight\n')
            f.write('0.00,90.00,0.00,1.00\n')
            f.close()
        model.nodal_plane_dist_filename='model01_focal_mech_dist.hdf5'
        #
        # saving the project
        oqtkp.models[model_id] = model
        oqtkp.save()

    def test_01(self):
        """
        This implements a workflow similar to the one used with FMG
        """
        reports_folder = './../tmp/project_test/reports/'
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
        nb_path = './../../notebooks/sources_area/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        nb.run(nb_full_path, '')
        #
        # .....................................................................
        # catalogue pre-processing 
        nb_name = 'catalogue_pre_processing.ipynb' 
        nb_path = './../../notebooks/catalogue/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        nb.run(nb_full_path, '')
        #
        # checking the creation of the pickled version of the catalogue
        file_name = 'model01_catalogue.pkl'
        file_path = './../tmp/project_test/'
        tmp = os.path.join(self.BASE_DATA_PATH, file_path, file_name)
        nb_full_path = os.path.abspath(tmp)
        assert os.path.exists(nb_full_path)
        #
        # checking that .hdf5 file exists and contains updated information
        file_name = 'completeness.hdf5'
        file_path = './../tmp/project_test/'
        tmp = os.path.join(self.BASE_DATA_PATH, file_path, file_name)
        nb_full_path = os.path.abspath(tmp)
        assert os.path.exists(nb_full_path)
        #
        # this is clearly non completely consistent. We should remove the 
        # duplicated thresholds and keep only the ones with the smaller
        # magnitude
        f = h5py.File(nb_full_path, 'r') 
        grp = f['/model01']
        computed = grp['whole_catalogue'][:]
        expected = np.array([[1998., 3.5], 
                             [1989., 4.0],
                             [1977., 4.5],
                             [1970., 5.0],
                             [1933., 5.5],
                             [1933., 6.0],
                             [1905., 6.5],
                             [1905., 7.0]])
        np.testing.assert_equal(expected, computed)
        f.close()
        #
        # .....................................................................
        # assign default completeness to all the sources
        nb_name = 'set_completeness_to_all_area_sources.ipynb' 
        nb_path = './../../notebooks/sources_area/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        nb.run(nb_full_path, '')
        #
        # checking that the .hdf5 contains the completeness tables for all the
        # sources 
        file_name = 'completeness.hdf5'
        file_path = './../tmp/project_test/'
        tmp = os.path.join(self.BASE_DATA_PATH, file_path, file_name)
        nb_full_path = os.path.abspath(tmp)
        f = h5py.File(nb_full_path, 'r') 
        grp = f['/model01']
        computed = grp['1'][:]
        np.testing.assert_equal(expected, computed)
        computed = grp['2'][:]
        np.testing.assert_equal(expected, computed)
        computed = grp['3'][:]
        np.testing.assert_equal(expected, computed)
        f.close()
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
        nb_path = './../../notebooks/sources_area/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        automator.run(self.prj_path, 'model01', nb_full_path, get_src_ids.keys,
                      reports_folder=reports_folder)
        #
        # loading the project
        del oqtkp
        oqtkp = OQtProject.load_from_file(self.prj_path)
        oqtkp.active_model_id = 'model01'
        model = oqtkp.models['model01']
        #
        # check the a and b values computed
        keys = list(model.sources.keys())
        self.assertAlmostEqual(model.sources['1'].a_gr, 3.7243511906)
        self.assertAlmostEqual(model.sources['1'].b_gr, 0.636452331875)
        self.assertAlmostEqual(model.sources['2'].a_gr, 3.69438318983)
        self.assertAlmostEqual(model.sources['2'].b_gr, 0.674434277192)
        self.assertAlmostEqual(model.sources['3'].a_gr, 3.32936780717)
        self.assertAlmostEqual(model.sources['3'].b_gr, 0.6336174742)
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
        nb_path = './../../notebooks/sources_area/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        automator.run(self.prj_path, 'model01', nb_full_path, get_src_ids.keys)
        #
        # checking that the .hdf5 contains the completeness tables for all the
        # sources 
        file_name = 'model01_hypo_dist.hdf5'
        file_path = './../tmp/project_test/'
        tmp = os.path.join(self.BASE_DATA_PATH, file_path, file_name)
        nb_full_path = os.path.abspath(tmp)
        assert os.path.exists(nb_full_path)
        # checking values
        expected = np.zeros(2, dtype=[('depth','f4'),('wei', 'f4')])
        expected[0] = (10.0, 0.6)
        expected[1] = (20.0, 0.4)
        f = h5py.File(nb_full_path, 'r')
        computed = f['1'][:]
        np.testing.assert_array_equal(expected, computed)
        computed = f['2'][:]
        np.testing.assert_array_equal(expected, computed)
        computed = f['3'][:]
        np.testing.assert_array_equal(expected, computed)
        f.close()
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
        nb_path = './../../notebooks/sources_area/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        automator.run(self.prj_path, 'model01', nb_full_path, get_src_ids.keys)
        #
        # checking that the .hdf5 contains the completeness tables for all the
        # sources 
        file_name = 'model01_focal_mech_dist.hdf5'
        file_path = './../tmp/project_test/'
        tmp = os.path.join(self.BASE_DATA_PATH, file_path, file_name)
        nb_full_path = os.path.abspath(tmp)
        assert os.path.exists(nb_full_path)
        # checking values
        expected = np.zeros(1, dtype=[('strike','f4'),('dip', 'f4'), ('rake', 'f4'), ('wei', 'f4')])
        expected[0] = (0.00,90.00,0.00,1.00)
        f = h5py.File(nb_full_path, 'r')
        computed = f['1'][:]
        np.testing.assert_array_equal(expected, computed)
        computed = f['2'][:]
        np.testing.assert_array_equal(expected, computed)
        computed = f['3'][:]
        np.testing.assert_array_equal(expected, computed)
        f.close()
        #
        # STRAIN ANALYSIS
        # .....................................................................
        # Computing moment from strain
        nb_name = 'compute_mo_from_strain.ipynb'
        nb_path = './../../notebooks/sources_area/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        nb.run(nb_full_path, '') 
        #
        # computing corner magnitude
        get_src_ids = GetSourceIDs(model)
        get_src_ids.keep_equal_to('source_type', ['AreaSource'])
        nb_name = 'compute_mc_from_mo.ipynb'
        nb_path = './../../notebooks/tectonics/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        automator.run(self.prj_path, 'model01', nb_full_path, get_src_ids.keys)
        # 
        # checking
        del oqtkp
        oqtkp = OQtProject.load_from_file(self.prj_path)
        oqtkp.active_model_id = 'model01'
        model = oqtkp.models['model01']
        thrs = 1e7 
        self.assertTrue(model.sources['1'].mo_mcs/8.2392996092e+15 < thrs)
        self.assertTrue(model.sources['2'].mo_mcs/1.99901877766e+16 < thrs)
        self.assertTrue(model.sources['3'].mo_mcs/1.99901877766e+16 < thrs)
        self.assertTrue(model.sources['1'].mo_strain/7.86150975109e+16 < thrs)
        self.assertTrue(model.sources['2'].mo_strain/5.29894154843e+16 < thrs)
        self.assertTrue(model.sources['3'].mo_strain/8.33252270107e+16 < thrs)
        #
        # fixing the MFD for all the area sources
        get_src_ids = GetSourceIDs(model)
        get_src_ids.keep_equal_to('source_type', ['AreaSource'])
        nb_name = 'set_mfd_tapered_GR.ipynb'
        nb_path = './../../notebooks/sources_area/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        automator.run(self.prj_path, 'model01', nb_full_path, get_src_ids.keys)
        #
        # FAULT SOURCES
        # .....................................................................
        # running the notebook that loads data from 
        nb_name = 'load_data_from_shapefile_fmg.ipynb'
        nb_path = './../../notebooks/sources_shallow_fault/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        nb.run(nb_full_path, '')
	#
        # checking the number of fault sources loaded
        del oqtkp
        oqtkp = OQtProject.load_from_file(self.prj_path)
        oqtkp.active_model_id = 'model01'
        model = oqtkp.models['model01']
        cnt = 0 
        for key in list(model.sources.keys()):
            src = model.sources[key]
            if src.source_type == 'SimpleFaultSource':
                cnt += 1
        assert cnt == 6
        #
        # compute the mfd from slip rate
        get_src_ids = GetSourceIDs(model)
        get_src_ids.keep_equal_to('source_type', ['SimpleFaultSource'])
        nb_name = 'mfd_double_truncated_from_slip_rate_SRC.ipynb'
        nb_path = './../../notebooks/sources_shallow_fault/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        automator.run(self.prj_path, 'model01', nb_full_path, get_src_ids.keys)
        #
        # checking that each fault has an MFD
        del oqtkp
        oqtkp = OQtProject.load_from_file(self.prj_path)
        oqtkp.active_model_id = 'model01'
        model = oqtkp.models['model01']
        for key in list(model.sources.keys()):
            src = model.sources[key]
            if src.source_type == 'SimpleFaultSource':
                assert hasattr(src, 'mfd') 
        #
        # .....................................................................
        # find the faults inside each area source
        get_src_ids.reset()
        get_src_ids = GetSourceIDs(model)
        get_src_ids.keep_equal_to('source_type', ['AreaSource'])
        nb_name = 'find_faults_within_area_source.ipynb'
        nb_path = './../../notebooks/sources_area/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        automator.run(self.prj_path, 'model01', nb_full_path, get_src_ids.keys)
        #
        # checking that each fault has an MFD
        del oqtkp
        oqtkp = OQtProject.load_from_file(self.prj_path)
        oqtkp.active_model_id = 'model01'
        model = oqtkp.models['model01']
        src = model.sources['1']
        self.assertAlmostEqual(src.ids_faults_inside['sf400'], 0.695494958) 
        self.assertAlmostEqual(src.ids_faults_inside['sf399'], 1.0) 
        src = model.sources['2']
        self.assertAlmostEqual(src.ids_faults_inside['sf398'], 1.0) 
        self.assertAlmostEqual(src.ids_faults_inside['sf396'], 1.0) 
        src = model.sources['3']
        self.assertAlmostEqual(src.ids_faults_inside['sf397'], 1.0) 
        self.assertAlmostEqual(src.ids_faults_inside['sf400'], 0.3045975665) 
        self.assertAlmostEqual(src.ids_faults_inside['sf395'], 0.2386801966) 
        #
        # .....................................................................
        # compute moment 
        nb_name = 'compute_mo_from_mfd.ipynb'
        nb_path = './../../notebooks/sources/'
        tmp = os.path.join(self.BASE_DATA_PATH, nb_path, nb_name)
        nb_full_path = os.path.abspath(tmp)
        nb.run(nb_full_path, '')
        # checking
        del oqtkp
        oqtkp = OQtProject.load_from_file(self.prj_path)
        oqtkp.active_model_id = 'model01'
        model = oqtkp.models['model01']
        for key in list(model.sources.keys()):
            src = model.sources[key]
            self.assertTrue(hasattr(src, 'mo_from_mfd'))
