# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

import os
import unittest
import numpy as np
import subprocess
import tempfile

from pathlib import Path
from glob import glob
from openquake.hazardlib.source import PointSource, SimpleFaultSource
from openquake.hazardlib.scalerel import PointMSR, WC1994
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.geo import Point, NodalPlane, Line
from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.sourcewriter import write_source_model

HERE = Path(__file__).parent


class ClipDSAroundFaultsTest(unittest.TestCase):

    def test_with_multipoint_sources(self):

        # Create the temporary output folder
        outdir = Path(tempfile.gettempdir())
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outdir = tempfile.mkdtemp(suffix=None, prefix='mpoint', dir=outdir)

        faults_fname = HERE / 'data' / 'multipoint' / 'src-fault_02.xml'
        mpoints_fname = HERE / 'data' / 'multipoint' / 'src-mpoints_01.xml'

        cmd = 'oqm wkf remove_buffer_around_faults '
        cmd += f'{faults_fname} {mpoints_fname} {outdir} 10 6.0'
        _ = subprocess.run(cmd, shell=True)

    def test_two_point_srcs_files_two_faults(self):
        """ Test clipping distributed seismicity around 2 faults """

        # Create the temporary folders
        tmpdir = Path(tempfile.gettempdir())
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        faults_dir = tempfile.mkdtemp(suffix='faults', prefix=None, dir=tmpdir)
        points_dir = tempfile.mkdtemp(suffix='points', prefix=None, dir=tmpdir)

        # Create the point sources
        trt = 'Active Shallow Crust'
        mfd = TruncatedGRMFD(min_mag=5., max_mag=7.5, bin_width=0.1, a_val=0.5,
                             b_val=1.)
        npd = PMF([(1, NodalPlane(strike=0., dip=50, rake=90.))])
        hpd = PMF([(0.2, 5.), (0.5, 10.), (0.3, 15.)])

        for ii in ['1', '2']:
            ya = -2
            lon = np.arange(-2, 2, 0.2)
            lat = np.arange(ya, ya + 2, 0.2)
            ya += 2
            sources = []
            for lo in lon:
                for la in lat:

                    src1 = PointSource(
                        source_id=ii, name='point{}'.format(ii),
                        tectonic_region_type=trt, mfd=mfd,
                        rupture_mesh_spacing=2., rupture_aspect_ratio=1.,
                        magnitude_scaling_relationship=PointMSR(),
                        temporal_occurrence_model=PoissonTOM(50.),
                        upper_seismogenic_depth=2.,
                        lower_seismogenic_depth=20.,
                        nodal_plane_distribution=npd,
                        hypocenter_distribution=hpd,
                        location=Point(lo, la))
                sources.append(src1)

            fmt = '{}/sources_{}.xml'
            write_source_model(fmt.format(points_dir, ii), sources)

        # then create the faults
        dips = [30, 90]
        rakes = [90, 0]
        ddw = [45, 20]
        usd = 0
        rms = 2
        rar = 1.0
        points = [[Point(0, 0), Point(0.3, 0.3)],
                  [Point(-1, -1), Point(-0.3, -0.3)]]
        mfds = [TruncatedGRMFD(min_mag=6.5, max_mag=7.5, bin_width=0.1,
                               a_val=5, b_val=1.),
                TruncatedGRMFD(min_mag=6.5, max_mag=8.0, bin_width=0.1,
                               a_val=6, b_val=1.)]
        trt = 'Active Shallow Crust'

        faults = []
        for ii, dip in enumerate(dips):
            src = SimpleFaultSource(
                fault_trace=Line(points[ii]),
                source_id=str(ii),
                name='sf{}'.format(ii),
                tectonic_region_type=trt,
                mfd=mfds[ii],
                rupture_mesh_spacing=rms,
                magnitude_scaling_relationship=WC1994(),
                rupture_aspect_ratio=rar,
                temporal_occurrence_model=PoissonTOM(1.),
                upper_seismogenic_depth=usd,
                lower_seismogenic_depth=ddw[ii] * np.sin(np.deg2rad(dip)),
                dip=dip, rake=rakes[ii])
            faults.append(src)

        faults_fname = '{}/faults.xml'.format(faults_dir)
        write_source_model(faults_fname, faults)

        # Clip the point sources around the faults
        points_fnames = points_dir + '/sources*.xml'
        folder_out = tempfile.mkdtemp(suffix='out', prefix=None, dir=tmpdir)
        cmd = "oqm wkf remove_buffer_around_faults "
        cmd += f"{faults_fname} '{points_fnames}' {folder_out} 10.0 6.5"
        _ = subprocess.run(cmd, shell=True)

        # Check if new files match the ones in expected
        files = glob(folder_out)
        files_e = glob('expected/ds_test_01')

        for f1, f2 in zip(files, files_e):
            self.assertEqualFiles(f1, f2)
