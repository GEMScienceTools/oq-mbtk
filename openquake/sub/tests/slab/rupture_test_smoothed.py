"""
Module :module:`openquake.sub.tests.slab.rupture_test_smoothed`
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

PLOTTING = True

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

        # Save the new .ini
        self.ini = os.path.join(self.out_path, 'test.ini')
        with open(self.ini, 'w') as configfile:
            config.write(configfile)
        self.config = config

        # Create the complex surface
        max_sampl_dist = 10.
        build_complex_surface(in_path, max_sampl_dist, self.out_path,
                              upper_depth=50, lower_depth=300)

        print(self.out_path)

    def tearDown(self):
        pass
        #shutil.rmtree(self.out_path)

    def test01(self):
        """ Test smoothing """

        # Create the ruptures
        reff = os.path.join(BASE_DATA_PATH, '..', 'data', 'ini')
        """
        """
        calculate_ruptures(self.ini, False, reff)

        # Create .xml with OQ input
        label = 'test'
        rupture_hdf5_fname = self.config['main']['out_hdf5_fname']
        investigation_t = '1.'
        create(label, rupture_hdf5_fname, self.out_path, investigation_t)

        # Read .xml and calculate the rates of occurrence within each
        # magnitude bin
        pattern = os.path.join(self.out_path, '*.nrml')
        rates = []
        for source_model_fname in sorted(glob.glob(pattern)):
            mag, rate = get_mags_rates(source_model_fname, 1.0)
            rates.append([mag, rate])
        print(rates)

        # Calculate the expected rates
        mags = np.arange(7.5, 8.01, 0.1)
        agr = float(self.config['main']['agr'])
        bgr = float(self.config['main']['bgr'])
        rates_gr = 10**(agr-bgr*mags[:-1]) - 10**(agr-bgr*mags[1:])

        print(rates_gr)

        if 1:

            # See https://docs.pyvista.org/user-guide/index.html# note also
            # that the zone crosses the IDL
            vscaling = -0.01

            import pyvista as pv

            plt_smooth = False
            plt_rup_wei = True

            plotter = pv.Plotter()
            #plotter.set_background('white')

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
            print(min(swe), max(swe))

            # Ruptures
            fname = os.path.join(self.out_path, '7.55.hdf5')
            f = h5py.File(fname, 'r')
            coo = f['src_test_7pt55']['hypocenter'][:]
            coo[coo[:, 0] < 0, 0] = coo[coo[:, 0] < 0, 0]+360
            coo[:, 2] *= vscaling
            prb = f['src_test_7pt55']['probs_occur'][:]
            print(prb.shape, coo.shape)
            f.close()

            # Plotting
            #mesh.plot(background='white', cpos='xy', cmap='plasma', show_scalar_bar=True)
            if plt_smooth:
                mesh = pv.PolyData(points)
                mesh['scalars'] = swe
                _ = plotter.add_mesh(mesh=mesh, cmap='jet', show_scalar_bar=True)
            if plt_rup_wei:
                mesh = pv.PolyData(coo)
                mesh['scalars'] = prb[:, 1]
                _ = plotter.add_mesh(mesh=mesh, cmap='jet', show_scalar_bar=True)
            _ = plotter.show(interactive=True)
