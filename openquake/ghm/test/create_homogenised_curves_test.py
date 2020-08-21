"""
Module create_map_test
"""

import os
import re
import shutil
import unittest
import numpy as np
import geopandas as gpd

from openquake.ghm.create_homogenised_curves import process_maps

DATA = os.path.join(os.path.dirname(__file__))


class CreateMapTestCase(unittest.TestCase):
    """ testing the calculation of hazard curves in case of a cluster """

    def setUp(self):
        self.outfolder = os.path.join(DATA, 'tmp')
        if not os.path.exists(self.outfolder):
            os.mkdir(self.outfolder)
            print('creating tmp')
        else:
            shutil.rmtree(self.outfolder)

    @unittest.skipUnless('GEMDATA' in os.environ, 'please set GEMDATA')
    def test_mean_file_search(self):
        """ Testing homogenisation between models"""

        # contacts_shp, outpath, datafolder, sidx_fname, shapefile

        # 1 - contacts shapefile
        fname = 'contacts_between_models.shp'
        contacts_shp = os.path.join(DATA, 'data', 'hom', 'gis', fname)
        # 2 - output folder
        outfname = self.outfolder
        # 3 - folder with data
        datafolder = os.path.join(DATA, 'data', 'hom', 'db')
        # 4 - folder with the spatial index
        fname = 'trigrd_split_9_spacing_13'
        sidx_fname = os.path.join(os.environ['GEMDATA'], 'global_grid', fname)
        # 5 - contacts shapefile
        fname = 'world_country_admin_boundary_with_fips_codes_mosaic_eu_russia.shp'
        shapefile = os.path.join(DATA, '..', 'data', 'gis', fname)
        # 6 - shapefile of inland areas
        fname = 'inland.shp'
        inland_shp = os.path.join(DATA, '..', 'data', 'gis', fname)
        # 7 - imt string
        imt_str = 'SA(0.1)'
        # 8 - keys of models used for the testing
        models = ['cca', 'sam']
        #
        process_maps(contacts_shp, outfname, datafolder, sidx_fname, shapefile,
                     imt_str, inland_shp, models)
        # read hazard curves in the buffer
        hcname = os.path.join(DATA, 'tmp', 'map_buffer.json')
        hcurves = gpd.read_file(hcname)
        # check results
        hazc = hcurves.iloc[33]
        poelabs = [l for l in hcurves.columns.tolist() if re.search('^poe', l)]
        expected = np.array([5.87959258e-01, 5.29447462e-01, 4.67035869e-01,
                             4.02760039e-01, 3.38808451e-01, 2.77377050e-01,
                             2.20493883e-01, 1.69819331e-01, 1.26483180e-01,
                             9.09924778e-02, 6.32168648e-02, 4.24462800e-02,
                             2.75691321e-02, 1.73115286e-02, 1.04678551e-02,
                             6.04183377e-03, 3.28685770e-03, 1.66196503e-03,
                             7.70176725e-04, 3.23170335e-04])
        computed = hazc[poelabs]
        np.testing.assert_almost_equal(computed, expected)
        # check results
        hazc = hcurves.iloc[246]
        poelabs = [l for l in hcurves.columns.tolist() if re.search('^poe', l)]
        expected = np.array([4.46833067e-01, 3.82865890e-01, 3.18697878e-01,
                             2.57426350e-01, 2.01706437e-01, 1.53404656e-01,
                             1.13414149e-01, 8.16993498e-02, 5.75268468e-02,
                             3.97570796e-02, 2.70851523e-02, 1.82278562e-02,
                             1.20709569e-02, 7.76381412e-03, 4.74724157e-03,
                             2.69125337e-03, 1.38178602e-03, 6.30190816e-04,
                             2.50503322e-04, 8.39700556e-05])
        computed = hazc[poelabs]
        np.testing.assert_almost_equal(computed, expected)
        # check results
        hazc = hcurves.iloc[287]
        poelabs = [l for l in hcurves.columns.tolist() if re.search('^poe', l)]
        expected = np.array([3.85157094e-01, 3.25494299e-01, 2.65799999e-01,
                             2.09199620e-01, 1.58596674e-01, 1.16044057e-01,
                             8.23761599e-02, 5.72214832e-02, 3.92698216e-02,
                             2.68427379e-02, 1.83047544e-02, 1.23414336e-02,
                             8.08399485e-03, 5.01907720e-03, 2.88174326e-03,
                             1.49444143e-03, 6.86701115e-04, 2.73956023e-04,
                             9.17899836e-05, 2.34512476e-05])
        computed = hazc[poelabs]
        np.testing.assert_almost_equal(computed, expected)
