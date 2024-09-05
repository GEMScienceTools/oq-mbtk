#!/usr/bin/env python
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

"""
Module :module:`openquake.sub.slab.rupture`
"""

import os
import re
import h5py
import numpy as np
# import pandas as pd
import rtree
import logging
import configparser
import pathlib

import matplotlib.pyplot as plt
from pyproj import Proj

# from mayavi import mlab
# from openquake.sub.plotting.tools import plot_mesh
# from openquake.sub.plotting.tools import plot_mesh_mayavi

from openquake.hazardlib.geo.surface.kite_fault import KiteSurface
from openquake.sub.quad.msh import create_lower_surface_mesh
from openquake.sub.grid3d import Grid3d
from openquake.sub.misc.profile import _read_profiles
from openquake.sub.misc.utils import (get_min_max, create_inslab_meshes,
                                      get_centroids)
from openquake.sub.slab.rupture_utils import (get_discrete_dimensions,
                                              get_ruptures, get_weights)

from openquake.baselib import sap
from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.scalerel import get_available_scalerel
from openquake.hmtk.seismicity.selector import CatalogueSelector
from openquake.hazardlib.geo.surface.gridded import GriddedSurface

from openquake.mbt.tools.smooth3d import Smoothing3D
from openquake.man.checks.catalogue import load_catalogue
from openquake.wkf.utils import create_folder

PLOTTING = FALSE


def get_catalogue(cat_pickle_fname, treg_filename=None, label='',
                  sort_cat=False):
    """
    :param cat_pickle_fname:
    :param treg_filename:
    :param label:
    """

    # Load TR
    if treg_filename is not None:
        f = h5py.File(treg_filename, 'r')
        tr = f[label][:]
        f.close()

    # Load the catalogue
    catalogue = load_catalogue(cat_pickle_fname)

    if sort_cat:
        catalogue.sort_catalogue_chronologically()

    # If a label and a TR are provided we filter the catalogue
    if treg_filename is not None:
        selector = CatalogueSelector(catalogue, create_copy=False)
        catalogue = selector.select_catalogue(tr)
    return catalogue


def smoothing(mlo, mla, mde, catalogue, hspa, vspa, fname):
    """
    :param mlo:
        Longitudes of the mesh
    :param mla:
        Latitudes of the mesh
    :param mde:
        Depths [downward positive] of the mesh
    :param catalogue:
        An earthquake catalogue instance in the hmtk format
    :param hspa:
        Grid horizontal spacing [in km]
    :param vspa:
        Grid vertical spacing [in km]
    :param fname:
        Name of the hdf5 where to store the results
    :returns:
        A tuple including the values on the grid provided and the smoothing
        object
    """

    # Create mesh with the 3D grid of points and complete the smoothing
    mesh = Mesh(mlo, mla, mde)

    # Smooth
    smooth = Smoothing3D(catalogue, mesh, hspa, vspa)
    values = smooth.gaussian(bffer=40, sigmas=[40, 20])

    # Output .hdf5 file
    if os.path.exists(fname):
        os.remove(fname)

    # Save results
    fh5 = h5py.File(fname, 'w')
    fh5.create_dataset('values', data=values)
    fh5.create_dataset('lons', data=mesh.lons)
    fh5.create_dataset('lats', data=mesh.lats)
    fh5.create_dataset('deps', data=mesh.depths)
    fh5.close()

    return values, smooth


def spatial_index(smooth):
    """
    Create a spatial index for the mesh used to smoooth the seismicity.

    :param smooth:
        An instance of the :class:`openquake.mbt.tools.smooth3d.Smoothing3D`
    :returns:
        A tuple containing the spatial index instance and the projection used
        to convert the geographic coordinates of the smoothing grid.
    """

    def _generator(mesh, p):
        """
        This is a generator function used to quickly populate the spatial
        index
        """
        for cnt, (lon, lat, dep) in enumerate(zip(mesh.lons.flatten(),
                                                  mesh.lats.flatten(),
                                                  mesh.depths.flatten())):
            x, y = p(lon, lat)
            x /= 1e3
            y /= 1e3
            yield (cnt, (x, y, dep, x, y, dep), 1)

    # Setting rtree properties
    prop = rtree.index.Property()
    prop.dimension = 3

    # Set the geographic projection
    lons = smooth.mesh.lons.flatten()
    p = Proj(proj='lcc', lon_0=np.mean(lons), lat_2=45)

    # Create the spatial index for the grid mesh
    r = rtree.index.Index(_generator(smooth.mesh, p), properties=prop)

    # Return the 3D spatial index - note that the index is in projected
    # coordinates
    return r, p


def create_ruptures(mfd, dips, sampling, msr, asprs, float_strike, float_dip,
                    r, values, oms, tspan, hdf5_filename, uniform_fraction,
                    proj, idl, align=False):
    """
    Create inslab ruptures using an MFD and a time span. The dictionary 'oms'
    contains lists of profiles for various values of dip. The ruptures are
    floated on each virtual fault created from a set of profiles.

    :param mfd:
        A magnitude frequency distribution
    :param dips:
        A set of dip values used to create the virtual faults within the slab.
    :param sampling:
        The distance in km used to sample the profiles
    :param msr:
        A magnitude scaling relationship instance
    :param asprs:
        A dictionary of aspect ratios (key: aspect ratio, value: weight)
    :param float_strike:
        Along strike rupture floating parameter
    :param float_dip:
        Along dip rupture floating parameter
    :param r:
        Spatial index for the nodes of the grid over which we smoothed
        seismicity
    :param values:
        Smothing results
    :param oms:
        A dictionary. Values of dip are used as keys while values of the
        dictionary are list of lists. Every list contains one or several
        :class:`openquake.hazardlib.geo.line.Line` instances each one
        corresponding to a 3D profile.
    :param tspan:
        Time span [in yr]
    :param hdf5_filename:
        Name of the hdf5 file where to store the ruptures
    :param uniform_fraction:
        Fraction of the overall rate for a given magnitude bin to be
        distributed uniformly to all the ruptures for the same mag bin.
    :param align:
        Profile alignment flag
    """

    # This contains (mag, depth) tuples indicating the depth below which
    # magnitudes lower than the threshold will be represented as a finite
    # rupture
    mag_dep_filter = np.array(
        [[6.0, 50.0],
         [7.0, 100.0]]
    )

    # Create the output hdf5 file
    fh5 = h5py.File(hdf5_filename, 'a')
    grp_inslab = fh5.create_group('inslab')

    # Loop over dip angles, traces on the top the slab surface and
    # magnitudes. The traces are used to create the virtual faults and
    # float the ruptures.
    allrup = {}
    iscnt = 0
    trup = 0
    for dip in dips:
        # Lines is a list of lines
        for mi, lines in enumerate(oms[dip]):

            logging.info(f'Virtual fault {mi} dip {dip:.2f}\n')

            # Filter out small virtual fault surfaces i.e. surfaces defined
            # by less than three profiles
            if len(lines) < 3:
                continue

            # Check initial profiles
            min_de = 1e10
            max_de = -1e10
            for lne in lines:
                ps = np.array([[p.longitude, p.latitude, p.depth] for p in
                               lne.points])
                min_de = np.min([min_de, np.min(ps[:, 2])])
                max_de = np.max([max_de, np.max(ps[:, 2])])
                assert not np.any(np.isnan(ps))

            # Create in-slab virtual fault - `lines` is the list of profiles
            # to be used for the construction of the virtual fault surface
            # 'smsh'
            ks = KiteSurface.from_profiles(
                lines, sampling, sampling, idl, align)

            # TODO This can probably be refactored since smsh is required only
            # when storing
            smsh = np.empty((ks.mesh.lons.shape[0], ks.mesh.lons.shape[1], 3))
            smsh[..., 0] = ks.mesh.lons
            smsh[..., 1] = ks.mesh.lats
            smsh[..., 2] = ks.mesh.depths

            # Store data describing the geometry of the virtual fault into the
            # hdf5 file
            grp_inslab.create_dataset('{:09d}'.format(iscnt), data=smsh)

            # Get centroids of ruptures occurring on the virtual fault surface
            ccc = get_centroids(smsh[:, :, 0], smsh[:, :, 1], smsh[:, :, 2])

            # Get weights - this assigns to each centroid the weight of
            # the closest node in the values array
            weights = get_weights(ccc, r, values, proj)

            # Loop over magnitudes
            for mag, _ in mfd.get_annual_occurrence_rates():

                # Here we decide if earthquakes with magnitude 'mag' must be
                # represented either as a finite or a point rupture
                as_point = False
                idx = np.argmax(mag < mag_dep_filter[:, 0])
                if min_de < mag_dep_filter[idx, 1]:
                    as_point = True

                # TODO this assigns arbitrarly a rake of 90 degrees. It
                # should be a configuration parameter
                area = msr.get_median_area(mag=mag, rake=90)
                rups = []
                for aspr in asprs:

                    # IMPORTANT: the sampling here must be consistent with
                    # what we use for the construction of the mesh
                    lng, wdt = get_discrete_dimensions(area, sampling, aspr)

                    # If one of the dimensions is equal to 0 it means that
                    # this aspect ratio cannot be represented with the value of
                    # sampling
                    if (lng is None or wdt is None or
                            lng < 1e-10 or wdt < 1e-10):
                        msg = f'Ruptures for mag {mag:.2f} and ar {aspr:.2f}'
                        msg = f'{msg} will not be defined'
                        logging.warning(msg)
                        continue

                    # Rupture lenght and rupture width as multiples of the
                    # mesh sampling distance
                    rup_len = int(lng / sampling) + 1
                    rup_wid = int(wdt / sampling) + 1

                    # Skip small ruptures
                    if rup_len < 2 or rup_wid < 2:
                        msg = 'Found a small rupture size'
                        logging.warning(msg)
                        continue

                    # Get Ruptures. f_strike and f_dip are the floating
                    # parameters along strike and dip. 'rup_len' and 'rup_wid'
                    # are in terms of number of cells
                    counter = 0
                    for rup, rl, cl in get_ruptures(ks.mesh, rup_len, rup_wid,
                                                    f_strike=float_strike,
                                                    f_dip=float_dip):

                        # Get the weight for this rupture from the smoothing
                        wsum = asprs[aspr]
                        wsum_smoo = np.nan

                        if uniform_fraction < 0.99:
                            w = weights[rl:rl + rup_wid - 1,
                                        cl:cl + rup_len - 1]
                            i = np.isfinite(w)
                            # Weight normalized by the number of points
                            # composing the mesh
                            tmpw = sum(w[i]) / len(i)
                            wsum_smoo = tmpw * asprs[aspr]

                        # Fix the longitudes outside the standard [-180, 180]
                        # range
                        ij = np.isfinite(rup[0])
                        iw = rup[0] > 180.
                        ik = np.logical_and(ij, iw)
                        rup[0][ik] -= 360

                        # Get centroid
                        idx_r = np.floor(rup[0].shape[0] / 2).astype('i4')
                        idx_c = np.floor(rup[0].shape[1] / 2).astype('i4')
                        hypo = [rup[0][idx_r, idx_c],
                                rup[1][idx_r, idx_c],
                                rup[2][idx_r, idx_c]]

                        # Checking
                        assert np.all(rup[0][ij] <= 180)
                        assert np.all(rup[0][ij] >= -180)

                        # Get coordinates of the rupture surface
                        rx = rup[0][ij].flatten()
                        ry = rup[1][ij].flatten()
                        rz = rup[2][ij].flatten()

                        # Create the gridded surface. We need at least four
                        # vertexes.
                        if len(rx) > 3:
                            srfc = GriddedSurface(Mesh.from_coords(zip(rx,
                                                                       ry,
                                                                       rz),
                                                                   sort=False))
                            # Update the list with the ruptures - the
                            # second-last element in the list is the container
                            # for the probability of occurrence. For the time
                            # being this is not defined. 'wsum' is the weight
                            # for the current aspect ratio, 'wsum_smoo' is the
                            # weight from the smoothing
                            rups.append([srfc, wsum, wsum_smoo, dip, aspr,
                                         [], hypo])
                            counter += 1
                            trup += 1

                # Update the list of ruptures
                lab = '{:.2f}'.format(mag)
                if lab in allrup:
                    allrup[lab] += rups
                else:
                    allrup[lab] = rups

            # Update counter
            iscnt += 1

    # Closing the hdf5 file
    fh5.close()

    # Logging info
    for lab in sorted(allrup.keys()):
        tmps = 'Number of ruptures for m={:s}: {:d}'
        logging.info(tmps.format(lab, len(allrup[lab])))

    # Compute the normalizing factor
    twei = {}
    tweis = {}
    for mag, occr in mfd.get_annual_occurrence_rates():
        smm = 0.
        smms = 0.
        lab = '{:.2f}'.format(mag)
        for _, wei, weis, _, _, _, _ in allrup[lab]:
            if np.isfinite(wei):
                smm += wei
            if np.isfinite(weis):
                smms += weis
        twei[lab] = smm
        tweis[lab] = smms
        tmps = 'Total weight {:s}: {:f}'
        logging.info(tmps.format(lab, twei[lab]))

    # Generate and store the final set of ruptures
    fh5 = h5py.File(hdf5_filename, 'a')
    grp_rup = fh5.create_group('ruptures')

    # Assign probability of occurrence
    for mag, occr in mfd.get_annual_occurrence_rates():

        # Create the label
        lab = '{:.2f}'.format(mag)

        # Check if weight is larger than 0
        if twei[lab] < 1e-50 and uniform_fraction < 0.99:
            tmps = 'Weight for magnitude {:s} equal to 0'
            tmps = tmps.format(lab)
            logging.warning(tmps)

        rups = []
        grp = grp_rup.create_group(lab)

        # Loop over the ruptures and compute the annual pocc
        cnt = 0
        chk = 0
        chks = 0
        for srfc, wei, weis, _, _, _, hypo in allrup[lab]:

            # Adjust the weight. Every rupture will have a weight that is
            # a combination between a flat rate and a spatially variable rate
            wei = wei / twei[lab]
            ocr = (occr * uniform_fraction) * wei
            chk += wei
            if uniform_fraction < 0.99:
                weis = weis / tweis[lab]
                ocr += (occr * (1. - uniform_fraction)) * weis
                chks += weis

            # Compute the probabilities
            p0 = np.exp(-ocr * tspan)
            p1 = 1. - p0

            # Append ruptures
            rups.append([srfc, [wei, weis], dip, aspr, [p0, p1]])

            # Preparing the data structure for storing information
            a = np.zeros(1, dtype=[('lons', 'f4', srfc.mesh.lons.shape),
                                   ('lats', 'f4', srfc.mesh.lons.shape),
                                   ('deps', 'f4', srfc.mesh.lons.shape),
                                   ('w', 'float32', (2)),
                                   ('dip', 'f4'),
                                   ('aspr', 'f4'),
                                   ('prbs', 'float32', (2)),
                                   ('hypo', 'float32', (3)),
                                   ])

            a['lons'] = srfc.mesh.lons
            a['lats'] = srfc.mesh.lats
            a['deps'] = srfc.mesh.depths
            a['w'] = [wei, weis]
            a['dip'] = dip
            a['aspr'] = aspr
            a['prbs'] = np.array([p0, p1], dtype='float32')
            a['hypo'] = hypo
            grp.create_dataset('{:08d}'.format(cnt), data=a)
            cnt += 1

        allrup[lab] = rups

        if len(rups):
            if uniform_fraction < 0.99:
                fmt = 'Sum of weights for smoothing: '
                fmt = '{:.5f}. Should be close to 1'
                msg = fmt.format(chks)
                assert (1.0 - chks) < 1e-5, msg

            if uniform_fraction > 0.01:
                fmt = 'Sum of weights for uniform: '
                fmt = '{:.5f}. Should be close to 1'
                msg = fmt.format(chk)
                assert (1.0 - chk) < 1e-5, msg

    fh5.close()

    return allrup


def list_of_floats_from_string(istr):
    """
    Return a list of floats included in a string
    """
    tstr = re.sub(r'(\[|\])', '', istr)
    return [float(d) for d in re.split(',', tstr)]


def dict_of_floats_from_string(istr):
    """
    Returns a dictionary from a string. Used to parse the config file.
    """
    tstr = re.sub(r'(\{|\})', '', istr)
    out = {}
    for tmp in re.split(',', tstr):
        elem = re.split(':', tmp)
        out[float(elem[0])] = float(elem[1])
    return out


# def calculate_ruptures(ini_fname, only_plt=False, ref_fdr=None, agr=None,
#                        bgr=None, mmin=None, mmax=None, **kwargs):
def calculate_ruptures(ini_fname, **kwargs):
    """
    Using the information in a configuration file,

    :param str ini_fname:
        The name of a .ini file
    :param only_plt:
        Boolean. When true only plots ruptures
    :param ref_fdr:
        The path to the reference folder used to set the paths in the .ini
        file. If not provided directly, we use the one set in the .ini file.
    """

    # Read config file
    config = configparser.ConfigParser()
    config.read_file(open(ini_fname))

    # Logging settings
    logging.basicConfig(format='rupture:%(levelname)s:%(message)s')

    # Reference folder
    if 'reference_folder' in kwargs:
        ref_fdr = kwargs.get('reference_folder')
    elif 'reference_folder' in config['main']:
        ref_fdr = config.get('main', 'reference_folder')
    else:
        ref_fdr = pathlib.Path(ini_fname).parent

    # Set parameters
    profile_sd_topsl = config.getfloat('main', 'profile_sd_topsl')
    edge_sd_topsl = config.getfloat('main', 'edge_sd_topsl')

    # Load parameters from the config file
    sampling = config.getfloat('main', 'sampling')
    float_strike = config.getfloat('main', 'float_strike')
    float_dip = config.getfloat('main', 'float_dip')
    slab_thickness = config.getfloat('main', 'slab_thickness')
    label = config.get('main', 'label')
    hspa = config.getfloat('main', 'hspa')
    vspa = config.getfloat('main', 'vspa')
    uniform_fraction = config.getfloat('main', 'uniform_fraction')

    # MFD params
    agr = kwargs.get('agr', config.getfloat('main', 'agr'))
    bgr = kwargs.get('bgr', config.getfloat('main', 'bgr'))
    mmax = kwargs.get('mmax', config.getfloat('main', 'mmax'))
    mmin = kwargs.get('mmin', config.getfloat('main', 'mmin'))

    # IDL
    idl = kwargs.get('idl', config.getboolean('main', 'idl', fallback=False))

    # Profile alignment at the top
    align = False
    if config.has_option('main', 'profile_alignment'):
        tmps = config.get('main', 'profile_alignment')
        if re.search('true', tmps.lower()):
            align = True

    # Set profile folder
    path = kwargs.get('profile_folder', config.get('main', 'profile_folder'))
    path = os.path.abspath(os.path.join(ref_fdr, path))

    # Catalogue
    cat_pickle_fname = config.get('main', 'catalogue_pickle_fname')
    cat_pickle_fname = os.path.abspath(os.path.join(ref_fdr, cat_pickle_fname))
    try:
        sort_cat = bool(config.get('main', 'sort_catalogue'))
    except Exception:
        sort_cat = False

    # Output
    hdf5_filename = kwargs.get('out_hdf5_fname',
                               config.get('main', 'out_hdf5_fname'))
    hdf5_filename = os.path.abspath(os.path.join(ref_fdr, hdf5_filename))

    # Smoothing output
    key = 'out_hdf5_smoothing_fname'
    out_hdf5_smoothing_fname = kwargs.get(key, config.get('main', key))
    tmps = os.path.join(ref_fdr, out_hdf5_smoothing_fname)
    out_hdf5_smoothing_fname = os.path.abspath(tmps)
    
    # create the smoothing directory if it doesn't exist
    smoothing_dir = os.path.sep.join(
        out_hdf5_smoothing_fname.split(os.path.sep)[:-1])
    if not os.path.exists(smoothing_dir):
        os.makedirs(smoothing_dir)

    # Tectonic regionalisation
    treg_filename = config.get('main', 'treg_fname')
    if not re.search('[a-z]', treg_filename):
        treg_filename = None
    else:
        treg_filename = os.path.abspath(os.path.join(ref_fdr, treg_filename))

    # Dip angles used to create the virtual faults within the slab
    dips = list_of_floats_from_string(config.get('main', 'dips'))
    asprsstr = config.get('main', 'aspect_ratios')
    asprs = dict_of_floats_from_string(asprsstr)

    # Magnitude-scaling relationship
    msrstr = config.get('main', 'mag_scaling_relation')
    msrd = get_available_scalerel()
    if msrstr not in msrd.keys():
        raise ValueError('')
    msr = msrd[msrstr]()

    logging.info('Reading profiles from: {:s}'.format(path))
    profiles, pro_fnames = _read_profiles(path)
    assert len(profiles) > 0

    # Create mesh from profiles
    logging.info('Creating top of slab mesh')
    mks = KiteSurface.from_profiles(
        profiles, profile_sd_topsl, edge_sd_topsl, idl)

    # TODO This can probably be refactored since smsh is required only
    # when storing
    msh = np.empty((mks.mesh.lons.shape[0], mks.mesh.lons.shape[1], 3))
    msh[..., 0] = mks.mesh.lons
    msh[..., 1] = mks.mesh.lats
    msh[..., 2] = mks.mesh.depths

    # Create inslab meshes. The output (i.e ohs) is a dictionary with the
    # values of dip as keys. The values in the dictionary
    # are :class:`openquake.hazardlib.geo.line.Line` instances
    logging.info('Creating ruptures on virtual faults')
    ohs = create_inslab_meshes(msh, dips, slab_thickness, sampling)


    if False:
        from mayavi import mlab

        # TODO consider replacing with pyvista
        azim = 10.
        elev = 20.
        dist = 20.

        f = mlab.figure(bgcolor=(1, 1, 1), size=(900, 600))
        vsc = -0.01

        # Profiles
        for ipro, (pro, fnme) in enumerate(zip(profiles, pro_fnames)):
            tmp = [[p.longitude, p.latitude, p.depth] for p in pro.points]
            tmp = np.array(tmp)
            tmp[tmp[:, 0] < 0, 0] = tmp[tmp[:, 0] < 0, 0] + 360
            mlab.plot3d(tmp[:, 0], tmp[:, 1], tmp[:, 2] * vsc, color=(1, 0, 0))

        # Top of the slab mesh
        # plot_mesh_mayavi(msh, vsc, color=(0, 1, 0))

        for key in ohs:
            for iii in range(len(ohs[key])):
                for line in ohs[key][iii]:
                    pnt = np.array([[p.longitude, p.latitude, p.depth]
                                    for p in line.points])
                    pnt[pnt[:, 0] < 0, 0] = pnt[pnt[:, 0] < 0, 0] + 360
                    mlab.plot3d(pnt[:, 0], pnt[:, 1], pnt[:, 2] * vsc,
                                color=(0, 0, 1))

        f.scene.camera.azimuth(azim)
        f.scene.camera.elevation(elev)
        mlab.view(distance=dist)
        mlab.show()
        mlab.show()
        exit(0)

    if PLOTTING:
        vsc = 0.01
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        #
        # profiles
        for ipro, (pro, fnme) in enumerate(zip(profiles, pro_fnames)):
            tmp = [[p.longitude, p.latitude, p.depth] for p in pro.points]
            tmp = np.array(tmp)
            tmp[tmp[:, 0] < 0, 0] = tmp[tmp[:, 0] < 0, 0] + 360
            ax.plot(tmp[:, 0], tmp[:, 1], tmp[:, 2] * vsc, 'x--b', ms=2)
            tmps = '{:d}-{:s}'.format(ipro, os.path.basename(fnme))
            ax.text(tmp[0, 0], tmp[0, 1], tmp[0, 2] * vsc, tmps)

        # Top of the slab mesh
        # plot_mesh(ax, msh, vsc)

        for key in ohs:
            for iii in range(len(ohs[key])):
                for line in ohs[key][iii]:
                    pnt = np.array([[p.longitude, p.latitude, p.depth]
                                    for p in line.points])
                    pnt[pnt[:, 0] < 0, 0] = pnt[pnt[:, 0] < 0, 0] + 360
                    ax.plot(pnt[:, 0], pnt[:, 1], pnt[:, 2]*vsc, '-r')
        ax.invert_zaxis()
        ax.view_init(50, 55)
        plt.show()

    # The one created here describes the bottom of the slab
    lmsh = create_lower_surface_mesh(msh, slab_thickness)

    # Get min and max values of the mesh
    milo, mila, mide, malo, mala, made = get_min_max(msh, lmsh)

    # Create the 3D mesh describing the volume of the slab. This `dlt` value
    # [in degrees] is used to create a buffer around the mesh
    dlt = 5.0

    msh3d = Grid3d(milo - dlt, mila - dlt, mide,
                   malo + dlt, mala + dlt, made, hspa, vspa)

    # Create three vectors with the coordinates of the nodes describing the
    # slab volume
    mlo, mla, mde = msh3d.get_coordinates_vectors()

    if False:
        import pandas as pd
        df = pd.DataFrame({'mlo': mlo, 'mla': mla, 'mde': mde})
        df.to_csv('mesh_coords.csv')

    # Removing pre-exising hdf5 file
    if os.path.exists(hdf5_filename):
        os.remove(hdf5_filename)
    else:
        path = os.path.dirname(hdf5_filename)
        create_folder(path)

    # Save data on hdf5 file
    logging.info('Creating {:s}'.format(hdf5_filename))
    fh5 = h5py.File(hdf5_filename, 'w')
    grp_slab = fh5.create_group('slab')
    dset = grp_slab.create_dataset('top', data=msh)
    dset.attrs['spacing'] = sampling
    grp_slab.create_dataset('bot', data=lmsh)
    fh5.close()

    # Get earthquake catalogue
    catalogue = get_catalogue(cat_pickle_fname, treg_filename, label,
                              sort_cat)

    # Smooth the seismicity within the volume of the slab
    values, smooth = smoothing(mlo, mla, mde, catalogue, hspa, vspa,
                               out_hdf5_smoothing_fname)

    # Create the spatial index
    r, proj = spatial_index(smooth)

    # Define the magnitude-frequency distribution
    mfd = TruncatedGRMFD(min_mag=mmin, max_mag=mmax, bin_width=0.1,
                         a_val=agr, b_val=bgr)

    # Create all the ruptures - the probability of occurrence is for one year
    # in this case
    _ = create_ruptures(mfd, dips, sampling, msr, asprs, float_strike,
                        float_dip, r, values, ohs, 1., hdf5_filename,
                        uniform_fraction, proj, idl, align)


calculate_ruptures.ini_fname = '.ini filename'
calculate_ruptures.only_plt = 'Only plotting'
calculate_ruptures.ref_fdr = 'Reference folder for paths'

if __name__ == "__main__":
    sap.run(calculate_ruptures)
