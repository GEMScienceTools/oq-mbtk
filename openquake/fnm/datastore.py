# ------------------- The OpenQuake Model Building Toolkit --------------------
# ------------------- FERMI: Fault nEtwoRks ModellIng -------------------------
# Copyright (C) 2023 GEM Foundation
#         .-.
#        /    \                                        .-.
#        | .`. ;    .--.    ___ .-.     ___ .-. .-.   ( __)
#        | |(___)  /    \  (   )   \   (   )   '   \  (''")
#        | |_     |  .-. ;  | ' .-. ;   |  .-.  .-. ;  | |
#       (   __)   |  | | |  |  / (___)  | |  | |  | |  | |
#        | |      |  |/  |  | |         | |  | |  | |  | |
#        | |      |  ' _.'  | |         | |  | |  | |  | |
#        | |      |  .'.-.  | |         | |  | |  | |  | |
#        | |      '  `-' /  | |         | |  | |  | |  | |
#       (___)      `.__.'  (___)       (___)(___)(___)(___)
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


import re
import h5py
import pathlib
import itertools
import numpy as np

from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.geo.surface.kite_fault import KiteSurface

KEYS = {
    'ruptures_single_section_indexes': 'section_idxs_per_rupture',
    'magnitudes': 'magnitudes',
    'ruptures_single_section': 'single_section_ruptures',
    'fault_system': {},
    'rupture_fractional_area': 'fraction_tot_area',
    'ruptures_indexes_of_sections_involved': 'section_idxs_per_rupture',
    'ruptures_connection_distances': 'conn_dists_per_rupture',
    'ruptures_connection_angles': 'conn_angles_per_rupture'
}


def _write_single_section_ruptures(fout, rups):
    """Write information on single section ruptures"""
    _ = fout.create_dataset("single_section_ruptures", data=rups)


def _write_ruptures(fout, rups):
    """
    Write rupture information to the datastore. Each rupture is described in
    terms of single section ruptures.
    """
    data = np.array(list(itertools.zip_longest(*rups, fillvalue=-1))).T
    _ = fout.create_dataset("ruptures", data=data)


def _write_magnitudes(fout, mags):
    """Write the value of magnitude for each rupture"""
    _ = fout.create_dataset("magnitudes", data=mags)


def _write_fault_system(fout, fsys):
    """Write fault system info"""
    fout.create_group("fault_system")
    fout.create_group("fault_system/meshes")
    fout.create_group("fault_system/rups")
    for i_sec, sec in enumerate(fsys):
        key = f"fault_system/meshes/mesh{i_sec:04d}"
        _ = fout.create_dataset(key, data=sec[0].mesh.array)
        key = f"fault_system/rups/rups{i_sec:04d}"
        _ = fout.create_dataset(key, data=sec[1])


def _write_fraction_area(fout, fracts):
    """
    Write the fraction of the total rupture on each section
    """
    data = np.array(list(itertools.zip_longest(*fracts, fillvalue=-1))).T
    _ = fout.create_dataset("fraction_tot_area", data=data)


def _write_rup_section_indexes(fout, idxs):
    """
    Write the index of the sections composing each rupture
    """
    data = np.array(list(itertools.zip_longest(*idxs, fillvalue=-1))).T
    _ = fout.create_dataset("section_idxs_per_rupture", data=data)


def _write_distances(fout, idxs):
    """
    Write the angles between the connections for each rupture
    """
    data = np.array(list(itertools.zip_longest(*idxs, fillvalue=-1))).T
    _ = fout.create_dataset("conn_dists_per_rupture", data=data)


def _write_angles(fout, idxs):
    """
    Write the angles between the connections for each rupture
    """
    data = np.array(list(itertools.zip_longest(*idxs, fillvalue=-1))).T
    _ = fout.create_dataset("conn_angles_per_rupture", data=data)


def write(fname, out):
    """
    Create the datastore with the information on ruptures

    :param fname:
        A string with the name of the .hdf5 file with the datastore
    :param out:
        A dictionary with the following keys:
            - ruptures_single_section_indexes
            - magnitudes
            - ruptures_single_section
            - fault_system
            - rupture_fractional_area
            - ruptures_indexes_of_sections_involved
            - ruptures_connection_distances
            - ruptures_connection_angles
    """

    # Get a pathlib Path instance
    if isinstance(fname, str):
        fname = pathlib.Path(fname)

    # Checking
    _check_output(out)

    # Create the folder if it does not exists
    fname.parents[0].mkdir(parents=True, exist_ok=True)

    # Open the .hdf5 file
    fout = h5py.File(str(fname), "w")

    # Write ruptures. Every row contains a list of single rupture indexes
    # N.B. discard the -1. These indexes correspond to the fifth column in
    # the dataset `single_section_ruptures`
    _write_ruptures(fout, out['ruptures_single_section_indexes'])

    # Write magnitudes
    _write_magnitudes(fout, out['magnitudes'])

    # Write single-section ruptures. For a description of the format read
    # :function:`openquake.fnm.rupture.get_ruptures_section`
    _write_single_section_ruptures(fout, out['ruptures_single_section'])

    # Write fault system info. For a description of the format check
    # :function:`openquake.fnm.fault_system.get_fault_system`
    _write_fault_system(fout, out['fault_system'])

    # Write fraction of total area
    _write_fraction_area(fout, out['rupture_fractional_area'])

    # Write indexes of sections composing each rupture. Each row contains
    # the indexes of the sections contributing to a rupture.
    key = 'ruptures_indexes_of_sections_involved'
    _write_rup_section_indexes(fout, out[key])

    # Write the distances and angles between connections
    _write_distances(fout, out['ruptures_connection_distances'])
    _write_angles(fout, out['ruptures_connection_angles'])

    print(fname)
    fout.close()


def _check_output(out):

    num_dists = len(out['ruptures_connection_distances'])
    num_angles = len(out['ruptures_connection_angles'])
    num_rups = len(out['ruptures_single_section_indexes'])
    assert num_dists == num_angles
    assert num_dists == num_rups


def _read_fault_system(fin, out):
    """Read fault system info"""

    try:

        fsys = []
        for key in fin['fault_system/meshes'].keys():

            fid = re.sub('[a-zA-Z]', '', key)

            # Get the mesh
            mesh_array = fin[f'fault_system/meshes/{key}'][:]
            mesh = Mesh(mesh_array[0], mesh_array[1], mesh_array[2])
            sfc = KiteSurface(mesh)

            # Get single-section ruptures
            rups_array = fin[f'fault_system/rups/rups{fid}'][:]
            fsys.append([sfc, rups_array])

    except ValueError as error:

        print(error)
        fin.close()

    out['fault_system'] = fsys
    return out


def read(fname):
    """
    :param fname:
        A string with the path to the .hdf5 file containing the datastore
    """
    out = {}

    # Open the .hdf5 file
    fin = h5py.File(str(fname), "r")

    # Load data
    for key in KEYS:
        try:
            if isinstance(KEYS[key], str):
                out[key] = fin[KEYS[key]][:]
            else:
                if key == 'fault_system':
                    out = _read_fault_system(fin, out)

        except ValueError as error:
            print(error)
            fin.close()

    fin.close()
    return out
