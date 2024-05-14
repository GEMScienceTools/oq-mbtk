# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2024 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
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
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import os
import unittest
import numpy as np
import subprocess
import tempfile
import pathlib
import filecmp
import shutil

from pathlib import Path
from glob import glob

from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.sourceconverter import SourceConverter

from openquake.hazardlib.source import (PointSource, SimpleFaultSource)
from openquake.hazardlib.scalerel import PointMSR, WC1994
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.geo import Point, NodalPlane, Line
from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.sourcewriter import write_source_model

HERE = Path(__file__).parent
PLOTTING = False
OVERWRITE = False


class ClipDSAroundFaultsTest(unittest.TestCase):

    def test_with_multipoint_sources(self):

        # Create the temporary output folder
        outdir = Path(tempfile.gettempdir())
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outdir = tempfile.mkdtemp(suffix=None, prefix='mpoint', dir=outdir)
        outdir = pathlib.Path(outdir)

        faults_fname = HERE / 'data' / 'multipoint' / 'src-fault_02.xml'
        mpoints_fname = HERE / 'data' / 'multipoint' / 'src-mpoints_01.xml'

        cmd = 'oqm wkf remove_buffer_around_faults '
        cmd += f'{faults_fname} {mpoints_fname} {outdir} 10 6.0'
        _ = subprocess.run(cmd, shell=True)

        # Check if new files match the ones in expected
        expc_dir = pathlib.Path('./expected/ds_test_02')
        files = sorted([f for f in outdir.glob('*.xml')])
        if OVERWRITE:
            shutil.copyfile(files[0], HERE / expc_dir / 'src_buffers_01.xml')
            shutil.copyfile(files[1], HERE / expc_dir / 'src_points_01.xml')
        files_e = sorted([f for f in expc_dir.glob('*.xml')])

        # Compare files
        for f1, f2 in zip(files, files_e):
            assert filecmp.cmp(f1, f2, shallow=True)


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
        folder_out = pathlib.Path(folder_out)
        cmd = "oqm wkf remove_buffer_around_faults "
        cmd += f"{faults_fname} '{points_fnames}' {folder_out} 10.0 6.5"
        _ = subprocess.run(cmd, shell=True)

        # Check if new files match the ones in expected
        expc_dir = pathlib.Path('./expected/ds_test_02')
        files = sorted([f for f in folder_out.glob('*.xml')])
        if OVERWRITE:
            shutil.copyfile(files[0], HERE / expc_dir / 'src_buffers_01.xml')
            shutil.copyfile(files[1], HERE / expc_dir / 'src_buffers_02.xml')
            shutil.copyfile(files[2], HERE / expc_dir / 'src_points_01.xml')
            shutil.copyfile(files[3], HERE / expc_dir / 'src_points_02.xml')
        files_e = sorted([f for f in expc_dir.glob('*.xml')])

        for f1, f2 in zip(files, files_e):
            assert filecmp.cmp(f1, f2, shallow=True)


    def test_area_sources(self):

        # Clip the area sources around the faults
        area_fnames = HERE / 'data' / 'areas_and_faults' / 'area_01.xml'
        faults_fname = HERE / 'data' / 'areas_and_faults' / 'fault_01.xml'

        # Output folder
        tmpdir = Path(tempfile.gettempdir())
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        folder_out = tempfile.mkdtemp(suffix='out', prefix=None, dir=tmpdir)
        folder_out = pathlib.Path(folder_out)

        # Running the code that merges the sources 
        cmd = "oqm wkf remove_buffer_around_faults "
        cmd += f"{faults_fname} '{area_fnames}' {folder_out} 10.0 6.5"
        out = subprocess.run(cmd, shell=True)

        # Check if the execution was successful
        msg = f'Execution was not successful {out}'
        self.assertTrue(out.returncode == 0, msg)

        print(f'Output stored in: {tmpdir}')

        # Plotting 
        fname_points = folder_out / 'src_points_01.xml'
        fname_buffer = folder_out / 'src_buffers_01.xml'

        if PLOTTING:
            plot_results(faults_fname, fname_points, fname_buffer)

        # Check if new files match the ones in expected
        expc_dir = pathlib.Path('./expected/ds_test_02')
        files = sorted([f for f in folder_out.glob('*.xml')])
        if OVERWRITE:
            shutil.copyfile(files[0], HERE / expc_dir / 'src_buffers_01.xml')
            shutil.copyfile(files[1], HERE / expc_dir / 'src_points_01.xml')
        files_e = sorted([f for f in expc_dir.glob('*.xml')])

        for f1, f2 in zip(files, files_e):
            assert filecmp.cmp(f1, f2, shallow=True)

    def test_area_sources_multif(self):

        # Clip the area sources around the faults
        area_fnames = HERE / 'data' / 'areas_and_faults' / 'area_01.xml'
        faults_fname = HERE / 'data' / 'areas_and_faults' / 'ruptures_0.xml'

        # Output folder
        tmpdir = Path(tempfile.gettempdir())
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        folder_out = tempfile.mkdtemp(suffix='out', prefix=None, dir=tmpdir)
        folder_out = pathlib.Path(folder_out)

        # Running the code that merges the sources 
        cmd = "oqm wkf remove_buffer_around_faults "
        cmd += f"{faults_fname} '{area_fnames}' {folder_out} 10.0 6.5"
        out = subprocess.run(cmd, shell=True)

        # Check if the execution was successful
        msg = f'Execution was not successful {out}'
        self.assertTrue(out.returncode == 0, msg)

        print(f'Output stored in: {tmpdir}')

        # Plotting 
        fname_points = folder_out / 'src_points_01.xml'
        fname_buffer = folder_out / 'src_buffers_01.xml'

        if PLOTTING:
            plot_results(faults_fname, fname_points, fname_buffer)

        # Check if new files match the ones in expected
        expc_dir = pathlib.Path('./expected/ds_test_02')
        files = sorted([f for f in folder_out.glob('*.xml')])
        if OVERWRITE:
            shutil.copyfile(files[0], HERE / expc_dir / 'src_buffers_01.xml')
            shutil.copyfile(files[1], HERE / expc_dir / 'src_points_01.xml')
        files_e = sorted([f for f in expc_dir.glob('*.xml')])

        for f1, f2 in zip(files, files_e):
            assert filecmp.cmp(f1, f2, shallow=True)


def plot_results(fname_faults, fname_points, fname_buffer):

    sconv = SourceConverter(
        investigation_time=1.0,
        rupture_mesh_spacing=5.0,
        complex_fault_mesh_spacing=10.0,
        width_of_mfd_bin=0.1,
        area_source_discretization=5.0)

    # Point sources outside the buffer
    ssm = to_python(fname_points, sconv)
    data = []
    for grp in ssm:
        for srcs in grp:
            for src in srcs:
                mmin, mmax = src.mfd.get_min_max_mag()
                data.append([src.location.longitude,
                             src.location.latitude,
                             mmax])
    data = np.array(data)

    # Point sources inside buffers
    buff = []
    ssm = to_python(fname_buffer, sconv)
    for grp in ssm:
        for srcs in grp:
            for src in srcs:
                mmin, mmax = src.mfd.get_min_max_mag()
                buff.append([src.location.longitude,
                             src.location.latitude,
                             mmax])
    buff = np.array(buff)

    import pygmt
    dlt = 0.1
    pygmt.makecpt(cmap="jet", series=[6.4, 7.4], continuous=False)
    fig = pygmt.Figure()
    fig.coast(region=[np.min(data[:, 0]),
                      np.max(data[:, 0]),
                      np.min(data[:, 1]),
                      np.max(data[:, 1])], shorelines=True, frame="a")

    # Plot Outside of buffer
    fig.plot(
        x=data[:, 0],
        y=data[:, 1],
        fill=data[:, 2],
        cmap=True,
        style="c0.3c",
        pen="black",
    )

    # Buffer
    fig.plot(
        x=buff[:, 0],
        y=buff[:, 1],
        fill=buff[:, 2],
        cmap=True,
        style="s0.3c",
        pen="red",
    )

    fig.colorbar(frame="xaf+lMax magnitude []")
    fig.show()
