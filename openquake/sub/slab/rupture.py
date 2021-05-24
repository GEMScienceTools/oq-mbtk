#!/usr/bin/env python
# coding: utf-8

"""
Module :module:`openquake.sub.slab.rupture`
"""


import os
import re
import h5py
import numpy as np
import rtree
import logging
import configparser

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from mayavi import mlab
from pyproj import Proj
# from openquake.sub.plotting.tools import plot_mesh
# from openquake.sub.plotting.tools import plot_mesh_mayavi

from openquake.sub.misc.edge import create_from_profiles
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

PLOTTING = False


def get_catalogue(cat_pickle_fname, treg_filename=None, label=''):
    """
    :param cat_pickle_fname:
    :param treg_filename:
    :param label:
    """
    #
    # loading TR
    if treg_filename is not None:
        f = h5py.File(treg_filename, 'r')
        tr = f[label][:]
        f.close()
    #
    # loading the catalogue
    # catalogue = pickle.load(open(cat_pickle_fname, 'rb'))
    catalogue = load_catalogue(cat_pickle_fname)
    catalogue.sort_catalogue_chronologically()
    #
    # if a label and a TR are provided we filter the catalogue
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
                    proj, idl, align=False, inslab=False):
    """
    Create inslab ruptures using an MFD, a time span. The dictionary 'oms'
    contains lists of profiles for various values of dip. The ruptures are
    floated on each virtual fault created from a set of profiles.

    :param mfd:
        A magnitude frequency distribution
    :param dips:
        A set of dip values used to create the virtual faults withni the slab.
    :param sampling:
        The distance in km used to
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

    # Create the output hdf5 file
    fh5 = h5py.File(hdf5_filename, 'a')
    grp_inslab = fh5.create_group('inslab')

    # Loop over dip angles, top traces on the top the slab surface and
    # magnitudes. The traces are used to create the virtual faults and
    # float the ruptures.
    allrup = {}
    iscnt = 0
    for dip in dips:
        for mi, lines in enumerate(oms[dip]):

            # Filter out small surfaces i.e. surfaces defined by less than
            # three profiles
            if len(lines) < 3:
                continue

            # Checking initial profiles
            for lne in lines:
                ps = np.array([[p.longitude, p.latitude, p.depth] for p in
                               lne.points])
                assert not np.any(np.isnan(ps))

            # Create in-slab virtual fault - `lines` is the list of profiles
            # to be used for the construction of the virtual fault surface
            smsh = create_from_profiles(lines, sampling, sampling, idl, align)

            # Create mesh
            omsh = Mesh(smsh[:, :, 0], smsh[:, :, 1], smsh[:, :, 2])

            # Store data in the hdf5 file
            grp_inslab.create_dataset('{:09d}'.format(iscnt), data=smsh)

            # Get centroids for a given virtual fault surface
            ccc = get_centroids(smsh[:, :, 0], smsh[:, :, 1], smsh[:, :, 2])

            # Get weights - this assigns to each cell centroid the weight of
            # the closest node in the values array
            weights = get_weights(ccc, r, values, proj)

            # Loop over magnitudes
            for mag, _ in mfd.get_annual_occurrence_rates():

                # TODO this is assigns arbitrarly a rake of 90 degrees. It
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
                        msg = 'Ruptures for magnitude {:.2f} and ar {:.2f}'
                        msg = msg.format(mag, aspr)
                        msg = '{:s} will not be defined'.format(msg)
                        logging.warning(msg)
                        continue

                    # Rupture lenght and rupture width as multiples of the
                    # mesh sampling distance
                    rup_len = int(lng/sampling) + 1
                    rup_wid = int(wdt/sampling) + 1

                    # Skip small ruptures
                    if rup_len < 2 or rup_wid < 2:
                        msg = 'Found an incompatible rupture size'
                        logging.warning(msg)
                        continue

                    # Get Ruptures
                    counter = 0
                    for rup, rl, cl in get_ruptures(omsh, rup_len, rup_wid,
                                                    f_strike=float_strike,
                                                    f_dip=float_dip):

                        # Get weights from the smoothing
                        tmpw = 1
                        if uniform_fraction > 0.99:
                            w = weights[cl:cl+rup_len-1, rl:cl+rup_wid-1]
                            i = np.isfinite(w)
                            tmpw = sum(w[i])

                        # Scale the weight using the aspect ratio weight
                        wsum = tmpw/asprs[aspr]

                        # Fix the longitudes outside the standard [-180, 180]
                        # range
                        ij = np.isfinite(rup[0])
                        iw = rup[0] > 180.
                        ik = np.logical_and(ij, iw)
                        rup[0][ik] -= 360

                        # Get centroid
                        idx_r = np.floor(rup[0].shape[0]/2).astype('i4')
                        idx_c = np.floor(rup[0].shape[1]/2).astype('i4')
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
                        # vertexes
                        if len(rx) > 3:
                            srfc = GriddedSurface(Mesh.from_coords(zip(rx,
                                                                       ry,
                                                                       rz),
                                                                   sort=False))
                            # Update the list with the ruptures - the last
                            # element in the list is the container for the
                            # probability of occurrence. For the time being
                            # this is not defined
                            rups.append([srfc, wsum, dip, aspr, [], hypo])
                            counter += 1

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

    # Compute the normalizing factor for every rupture. This is used only in
    # the case when smoothing is used a reference for distributing occurrence
    twei = {}
    for mag, occr in mfd.get_annual_occurrence_rates():
        smm = 0.
        lab = '{:.2f}'.format(mag)
        for _, wei, _, _, _, _ in allrup[lab]:
            if np.isfinite(wei):
                smm += wei
        twei[lab] = smm
        tmps = 'Total weight {:s}: {:f}'
        logging.info(tmps.format(lab, twei[lab]))

    # Generate and store the final set of ruptures
    fh5 = h5py.File(hdf5_filename, 'a')
    grp_rup = fh5.create_group('ruptures')

    # Assign probability of occurrence
    for mag, occr in mfd.get_annual_occurrence_rates():

        # Label
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
        for srfc, wei, _, _, _, hypo in allrup[lab]:

            # Adjust the weight. Every rupture will have a weight that is
            # a combination between a flat rate and a variable rate
            ocr = (occr * uniform_fraction) * wei
            if twei[lab] > 1e-10:
                wei /= twei[lab]
                ocr += (occr * (1.-uniform_fraction)) * wei
            chk += wei

            # Compute the probabilities
            p0 = np.exp(-ocr*tspan)
            p1 = 1. - p0

            # Append ruptures
            rups.append([srfc, wei, dip, aspr, [p0, p1]])

            # Preparing the data structure for storing information
            a = np.zeros(1, dtype=[('lons', 'f4', srfc.mesh.lons.shape),
                                   ('lats', 'f4', srfc.mesh.lons.shape),
                                   ('deps', 'f4', srfc.mesh.lons.shape),
                                   ('w', 'float32'),
                                   ('dip', 'f4'),
                                   ('aspr', 'f4'),
                                   ('prbs', 'float32', (2)),
                                   ('hypo', 'float32', (3)),
                                   ])

            a['lons'] = srfc.mesh.lons
            a['lats'] = srfc.mesh.lats
            a['deps'] = srfc.mesh.depths
            a['w'] = wei
            a['dip'] = dip
            a['aspr'] = aspr
            a['prbs'] = np.array([p0, p1], dtype='float32')
            a['hypo'] = hypo
            grp.create_dataset('{:08d}'.format(cnt), data=a)
            cnt += 1

        allrup[lab] = rups

        print('CHK', chk)

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


def calculate_ruptures(ini_fname, only_plt=False, ref_fdr=None):
    """
    :param str ini_fname:
        The name of a .ini file
    :param only_plt:
        Boolean. When true only it only plots ruptures
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
    if ref_fdr is None:
        if 'reference_folder' not in config['main']:
            msg = 'The .ini file does not contain the reference_folder param'
            raise ValueError(msg)
        ref_fdr = config.get('main', 'reference_folder')

    # Set parameters
    profile_sd_topsl = config.getfloat('main', 'profile_sd_topsl')
    edge_sd_topsl = config.getfloat('main', 'edge_sd_topsl')

    # This sampling distance is used to
    sampling = config.getfloat('main', 'sampling')
    float_strike = config.getfloat('main', 'float_strike')
    float_dip = config.getfloat('main', 'float_dip')
    slab_thickness = config.getfloat('main', 'slab_thickness')
    label = config.get('main', 'label')
    hspa = config.getfloat('main', 'hspa')
    vspa = config.getfloat('main', 'vspa')
    uniform_fraction = config.getfloat('main', 'uniform_fraction')

    # MFD params
    agr = config.getfloat('main', 'agr')
    bgr = config.getfloat('main', 'bgr')
    mmax = config.getfloat('main', 'mmax')
    mmin = config.getfloat('main', 'mmin')

    # IDL
    if config.has_option('main', 'idl'):
        idl = config.get('main', 'idl')
    else:
        idl = False

    # Profile aliggnment at the top
    align = False
    if config.has_option('main', 'profile_alignment'):
        tmps = config.get('main', 'profile_alignment')
        if re.search('true', tmps.lower()):
            align = True

    # Set profile folder
    path = config.get('main', 'profile_folder')
    path = os.path.abspath(os.path.join(ref_fdr, path))

    # Catalogue
    cat_pickle_fname = config.get('main', 'catalogue_pickle_fname')
    cat_pickle_fname = os.path.abspath(os.path.join(ref_fdr, cat_pickle_fname))

    # Output
    hdf5_filename = config.get('main', 'out_hdf5_fname')
    hdf5_filename = os.path.abspath(os.path.join(ref_fdr, hdf5_filename))

    # Smoothing output
    out_hdf5_smoothing_fname = config.get('main', 'out_hdf5_smoothing_fname')
    tmps = os.path.join(ref_fdr, out_hdf5_smoothing_fname)
    out_hdf5_smoothing_fname = os.path.abspath(tmps)

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
    print('>>', path)
    assert len(profiles) > 0

    # Create mesh from profiles
    msh = create_from_profiles(profiles, profile_sd_topsl, edge_sd_topsl, idl)

    # Create inslab meshes. The output (i.e ohs) is a dictionary with the
    # values of dip as keys. The values in the dictionary
    # are :class:`openquake.hazardlib.geo.line.Line` instances
    ohs = create_inslab_meshes(msh, dips, slab_thickness, sampling)

    if only_plt:
        pass

    # TODO consider replacing wiith pyvista
    """
        azim = 10.
        elev = 20.
        dist = 20.

        f = mlab.figure(bgcolor=(1, 1, 1), size=(900, 600))
        vsc = -0.01
        #
        # profiles
        for ipro, (pro, fnme) in enumerate(zip(profiles, pro_fnames)):
            tmp = [[p.longitude, p.latitude, p.depth] for p in pro.points]
            tmp = np.array(tmp)
            tmp[tmp[:, 0] < 0, 0] = tmp[tmp[:, 0] < 0, 0] + 360
            mlab.plot3d(tmp[:, 0], tmp[:, 1], tmp[:, 2]*vsc, color=(1, 0, 0))
        #
        # top of the slab mesh
        plot_mesh_mayavi(msh, vsc, color=(0, 1, 0))
        #
        for key in ohs:
            for iii in range(len(ohs[key])):
                for line in ohs[key][iii]:
                    pnt = np.array([[p.longitude, p.latitude, p.depth]
                                    for p in line.points])
                    pnt[pnt[:, 0] < 0, 0] = pnt[pnt[:, 0] < 0, 0] + 360
                    mlab.plot3d(pnt[:, 0], pnt[:, 1], pnt[:, 2]*vsc,
                                color=(0, 0, 1))

        f.scene.camera.azimuth(azim)
        f.scene.camera.elevation(elev)
        mlab.view(distance=dist)
        mlab.show()
        mlab.show()

        exit(0)
    """

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
            ax.plot(tmp[:, 0], tmp[:, 1], tmp[:, 2]*vsc, 'x--b', markersize=2)
            tmps = '{:d}-{:s}'.format(ipro, os.path.basename(fnme))
            ax.text(tmp[0, 0], tmp[0, 1], tmp[0, 2]*vsc, tmps)

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

    # Discretizing the slab
    # omsh = Mesh(msh[:, :, 0], msh[:, :, 1], msh[:, :, 2])
    # olmsh = Mesh(lmsh[:, :, 0], lmsh[:, :, 1], lmsh[:, :, 2])

    # this `dlt` value [in degrees] is used to create a buffer around the mesh
    dlt = 5.0
    msh3d = Grid3d(milo-dlt, mila-dlt, mide, malo+dlt, mala+dlt, made, hspa,
                   vspa)
    # mlo, mla, mde = msh3d.select_nodes_within_two_meshesa(omsh, olmsh)
    mlo, mla, mde = msh3d.get_coordinates_vectors()

    # save data on hdf5 file
    if os.path.exists(hdf5_filename):
        os.remove(hdf5_filename)
    logging.info('Creating {:s}'.format(hdf5_filename))
    fh5 = h5py.File(hdf5_filename, 'w')
    grp_slab = fh5.create_group('slab')
    dset = grp_slab.create_dataset('top', data=msh)
    dset.attrs['spacing'] = sampling
    grp_slab.create_dataset('bot', data=lmsh)
    fh5.close()

    # Get catalogue
    catalogue = get_catalogue(cat_pickle_fname, treg_filename, label)

    # smoothing
    values, smooth = smoothing(mlo, mla, mde, catalogue, hspa, vspa,
                               out_hdf5_smoothing_fname)

    # Spatial index
    r, proj = spatial_index(smooth)

    # magnitude-frequency distribution
    mfd = TruncatedGRMFD(min_mag=mmin, max_mag=mmax, bin_width=0.1,
                         a_val=agr, b_val=bgr)

    # Create all the ruptures - the probability of occurrence is for one year
    # in this case
    _ = create_ruptures(mfd, dips, sampling, msr, asprs, float_strike,
                        float_dip, r, values, ohs, 1., hdf5_filename,
                        uniform_fraction, proj, idl, align, True)


calculate_ruptures.ini_fname = '.ini filename'
calculate_ruptures.only_plt = 'Only plotting'
calculate_ruptures.ref_fdr = 'Reference folder for paths'

if __name__ == "__main__":
    sap.run(calculate_ruptures)
