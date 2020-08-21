import os
import numpy
import rtree
import rtree.index  # needed with version 0.9.4
import scipy.constants as consts

from openquake.mbt.tools.geo import get_idx_points_inside_polygon
from openquake.hazardlib.geo.geodetic import (point_at, geodetic_distance)


def check_idl(lons):
    idl = 0
    maxlon = max(lons)
    minlon = min(lons)
    if ((abs(maxlon - minlon) > 50) & ((maxlon / minlon) < 0)):
        idl = 1
    return idl


def coord_generators(mesh):
    for cnt, pnt in enumerate(mesh):
        idl = check_idl(mesh.lons)
        lon = pnt.longitude
        lat = pnt.latitude
        if idl == 1:
            lon = lon + 360 if lon < 0 else lon
        yield (cnt, (lon, lat, lon, lat), 1)


class Smoothing:
    """
    Class for smoothing a catalogue based on a set of gaussian
    smoothing kernels

    :parameter catalogue:
    :parameter mesh:
        An instance of :class:openquake.hazardlib
    :parameter cellsize:
    :parameter completeness:
    """

    def __init__(self, catalogue, mesh, cellsize, completeness=None):
        self.catalogue = catalogue
        self.mesh = mesh
        self.cellsize = cellsize
        self.completeness = completeness
        self._create_spatial_index()

    def _create_spatial_index(self):
        """
        This creates a rtree spatial index of the grid mesh.
        """
        # empty the tmp files
        tmp_file = ['./tmp.dat', './tmp.idx']
        for tmp in tmp_file:
            if os.path.exists(tmp):
                os.remove(tmp)
        # Create the spatial index for the grid mesh
        r = rtree.index.Index('./tmp')
        ids = set()
        for cnt, pnt in enumerate(coord_generators(self.mesh)):
            r.insert(id=pnt[0], coordinates=pnt[1])
            # Check that the point IDs are unique
            if pnt[0] not in ids:
                ids.add(pnt[0])
            else:
                print(pnt[0])
                raise ValueError('Index already used')
        # Set the index
        self.rtree = r

    def multiple_smoothing(self, params):
        """
        This performs a smoothing using a multiple kernels

        :parameter params:
            A list of tuples each one containing the name of the
            smoothing kernel, the required parameters and a weight.

            The supported smoothin kernels are the following ones:
            - 'gaussian' - The required parameters are the radius [km]
            and the standard deviation.

            Note that the sum of weights must be always sum to 1.

        :returns:
            An array
        """
        assert isinstance(params, list)
        valt = None
        for param in params:
            if param[0] == 'gaussian':
                val = self.gaussian(param[1], param[2])
                if valt is None:
                    valt = val * param[3]
                else:
                    valt += val * param[3]
        return valt

    def gaussian(self, radius, sigma):
        """
        :parameter radius:
            The maximum radius used [km]
        :parameter sigma:
            The standard deviation of the 2D gaussian kernel

        NOTE: this will not work across the IDL
        """
        # Values
        values = numpy.zeros((len(self.mesh)))
        # Smoothing the catalogue
        for lon, lat, mag in zip(self.catalogue.data['longitude'],
                                 self.catalogue.data['latitude'],
                                 self.catalogue.data['magnitude']):
            # check for idl and shift lon
            idl = check_idl(self.mesh.lons)
            if idl == 1:
                lon = lon + 360 if lon < 0 else lon
            # Set the bounding box
            minlon, minlat = point_at(lon, lat, 225, radius * 2 ** 0.5)
            maxlon, maxlat = point_at(lon, lat, 45, radius * 2 ** 0.5)
            # shift mins and maxs if idl
            if idl == 1:
                minlon = minlon + 360 if minlon < 0 else minlon
                maxlon = maxlon + 360 if maxlon < 0 else maxlon
            # find nodes within the bounding box
            idxs = list(set(self.rtree.intersection((minlon,
                                                     minlat,
                                                     maxlon,
                                                     maxlat))))
            # Get distances
            dsts = geodetic_distance(lon, lat,
                                     self.mesh.lons[idxs],
                                     self.mesh.lats[idxs])
            # Find indexes of nodes at a distance lower than the
            # radius
            jjj = numpy.nonzero(dsts < radius)[0]
            idxs = numpy.array(idxs)
            iii = idxs[jjj]
            # set values
            tmpvalues = numpy.exp(-dsts[jjj]**2/(2*sigma**2))
            # normalising
            normfact = sum(tmpvalues)
            values[iii] += tmpvalues/normfact
        return values

    def get_points_in_polygon(self, polygon):
        # first make idl adjustments
        idl = check_idl(self.mesh.lons)
        lons = polygon.lons
        if idl == 1:
            lons = [lon + 360 if lon < 0 else lon for lon in lons]

        minlon = min(polygon.lons)
        minlat = min(polygon.lats)
        maxlon = max(polygon.lons)
        maxlat = max(polygon.lats)

        idxs = list(self.rtree.intersection((minlon, minlat,
                                             maxlon, maxlat)))
        plons = self.mesh.lons[idxs]
        plats = self.mesh.lats[idxs]

        iii = get_idx_points_inside_polygon(plon=plons,
                                            plat=plats,
                                            poly_lon=polygon.lons,
                                            poly_lat=polygon.lats,
                                            pnt_idxs=idxs,
                                            buff_distance=0.)

        return iii
