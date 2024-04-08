"""
Module :module:`openquake.sub.tests.slab.rupture_smooth_test`
"""

import os
import glob
import h5py
import unittest
import shutil
import tempfile
import configparser
import numpy as np

from openquake.man.checks.rates import get_mags_rates
from openquake.sub.slab.rupture import calculate_ruptures, get_catalogue
from openquake.sub.create_inslab_nrml import create
from openquake.sub.build_complex_surface import build_complex_surface

PLOTTING = False
BASE_DATA_PATH = os.path.dirname(__file__)


class RuptureCreationSmoothedTest(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        We use the profiles used for the subduction in the model for the
        Pacific Islands to test the smoothing approach.
        """

        relpath = os.path.join('..', 'data', 'ini', 'test_kt_z1.ini')
        ini_fname = os.path.join(BASE_DATA_PATH, relpath)

        # Prepare the input folder and the output folder
        tmp = os.path.join('..', 'data', 'profiles', 'pai_kt_z1')
        in_path = os.path.join(BASE_DATA_PATH, tmp)

        # Create the tmp directory
        self.out_path = tempfile.mkdtemp()

        # Read the ini file and change params
        config = configparser.ConfigParser()
        config.read(ini_fname)
        tmp = os.path.join(self.out_path, 'ruptures.hdf5')
        config['main']['out_hdf5_fname'] = tmp
        tmp = os.path.join(self.out_path, 'smoothing.hdf5')
        config['main']['out_hdf5_smoothing_fname'] = tmp
        config['main']['profile_folder'] = self.out_path
        # Spatial distribution controlled by smoothing
        config['main']['uniform_fraction'] = '0.0'
        config['main']['sort_catalogue'] = 'True'

        # Save the new .ini
        self.ini = os.path.join(self.out_path, 'test.ini')
        with open(self.ini, 'w') as configfile:
            config.write(configfile)
        self.config = config

        # Create the complex surface
        max_sampl_dist = 10.
        build_complex_surface(in_path, max_sampl_dist, self.out_path,
                              upper_depth=50, lower_depth=300)

    def tearDown(self):
        shutil.rmtree(self.out_path)

    def test01(self):
        """ Test smoothing """

        # Create the ruptures
        self.reff = os.path.join(BASE_DATA_PATH, '..', 'data', 'ini')
        calculate_ruptures(self.ini, False, self.reff)

        # Create .xml with OQ input
        label = 'test'
        rupture_hdf5_fname = self.config['main']['out_hdf5_fname']
        investigation_t = '1.'
        create(label, rupture_hdf5_fname, self.out_path, investigation_t)

        # Read .xml and calculate the rates of occurrence within each
        # magnitude bin
        pattern = os.path.join(self.out_path, '*.xml')
        rates = []
        for source_model_fname in sorted(glob.glob(pattern)):
            mag, rate = get_mags_rates(source_model_fname, 1.0)
            rates.append([mag, rate])
        rates = np.array(rates)

        # Calculate the expected rates
        mags = np.arange(8.0, 8.21, 0.1)
        agr = float(self.config['main']['agr'])
        bgr = float(self.config['main']['bgr'])
        rates_gr = 10**(agr-bgr*mags[:-1]) - 10**(agr-bgr*mags[1:])

        np.testing.assert_almost_equal(rates[:, 1], rates_gr, decimal=3)

        if PLOTTING:
            # See https://docs.pyvista.org/user-guide/index.html# note also
            # that the zone crosses the IDL
            vscaling = -0.01

            import pyvista as pv

            plt_smooth = True
            plt_rup_wei = False

            plt_smooth = False
            plt_rup_wei = True

            plotter = pv.Plotter()
            plotter.set_background('grey')

            # Smoothing
            fname = os.path.join(self.out_path, 'smoothing.hdf5')
            f = h5py.File(fname, 'r')
            slo = f['lons'][:]
            slo[slo < 0] = slo[slo < 0]+360
            sla = f['lats'][:]
            sde = f['deps'][:] * vscaling
            swe = f['values'][:]
            f.close()
            points = np.array([slo, sla, sde]).T

            # Catalogue
            fname = os.path.join(self.reff,
                                 self.config['main']['catalogue_pickle_fname'])
            cat = get_catalogue(fname)
            lo = cat.data['longitude']
            cat.data['longitude'][lo < 0] = lo[lo < 0]+360
            flg = ((cat.data['longitude'] > min(points[:, 0])) &
                   (cat.data['longitude'] < max(points[:, 0])) &
                   (cat.data['latitude'] > min(points[:, 1])) &
                   (cat.data['latitude'] < max(points[:, 1])) &
                   (cat.data['depth'] > 30))
            catc = [(x, y, z) for x, y, z in zip(cat.data['longitude'][flg],
                    cat.data['latitude'][flg],
                    cat.data['depth'][flg] * vscaling)]
            catc = np.array(catc)

            # Ruptures
            fname = os.path.join(self.out_path, '8.05.hdf5')
            f = h5py.File(fname, 'r')
            coo = f['src_test_8pt05']['hypocenter'][:]
            coo[coo[:, 0] < 0, 0] = coo[coo[:, 0] < 0, 0]+360
            coo[:, 2] *= vscaling
            prb = f['src_test_8pt05']['probs_occur'][:]
            f.close()

            # Catalogue
            mesh = pv.PolyData(catc)
            _ = plotter.add_mesh(mesh=mesh, color='red',
                                 render_points_as_spheres=True)

            if plt_smooth:
                mesh = pv.PolyData(points)

                minval = 1e-7
                swe[swe < minval] = np.nan
                mesh['scalars'] = swe

                i = np.isfinite(swe)
                opac = np.zeros_like(swe)
                opac[i] = (swe[i] + 0.2 - minval) / max(swe[i])

                _ = plotter.add_mesh(mesh=mesh, cmap='jet',
                                     show_scalar_bar=True,
                                     point_size=7.0, nan_opacity=0.0,
                                     use_transparency=True, style='points',
                                     opacity=opac)

            if plt_rup_wei:
                mesh = pv.PolyData(coo)

                minval = 1e-7
                swe = prb[:, 1]
                swe[swe < minval] = np.nan
                mesh['scalars'] = swe
                _ = plotter.add_mesh(mesh=mesh, cmap='jet',
                                     show_scalar_bar=True, point_size=15.0,
                                     nan_opacity=0.0, opacity=1.0,
                                     style='points')

            _ = plotter.show(interactive=True)
