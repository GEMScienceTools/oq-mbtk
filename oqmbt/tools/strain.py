"""
"""

import numpy
import scipy.constants as constants

from oqmbt.tools.geo import get_idx_points_inside_polygon

from openquake.hazardlib.geo.polygon import Polygon
from openquake.hazardlib.source.area import AreaSource
from openquake.hazardlib.geo.utils import EARTH_RADIUS

def _get_moment_from_nodes(iii, strain_data, strain_idx, 
                           coupl_seism_thickness, shear_modulus,
                           cell_dx, cell_dy):
    """
    This function computes scalar seismic moment form a set of nodes of a 
    strain model

    :parameter iii:
        Indexes of cells in the strain model
    :parameter strain_data:
        A numpy array containing the strain rate model (the default format in 
        this case is the format used for the GSRM)
    :parameter strain_idx:
        An rtree instance for the strain rate model
    :parameter coupl_seism_thickness:
        A value of the coupled seismogenic thickness [km] i.e. the product 
        of the seismogenic thickness and a coupling factor [0, 1] 
    :parameter shear_modulus:
        A value of the shear modulus [Pa]
    """
    #
    # Check shear modulus
    assert (shear_modulus / 1e10) < 1e2
    #
    # Compute constant
    const = coupl_seism_thickness * shear_modulus
    #
    # Strain horizontal axess
    e1h = strain_data[iii,8]
    e2h = strain_data[iii,9]
    #
    # Get principal components
    evv = -(e1h+e2h)
    strain_values = numpy.column_stack([e1h, e2h, evv])
    e1rate = numpy.amin(strain_values, axis=1)
    e3rate = numpy.amax(strain_values, axis=1)
    e2rate = 0. - e1rate - e3rate
    # 
    # Cell area calculation: the default is 0.2 latitude x 0.25 longitude. 
    # I assume a spherical earth. Approximation should be reasonable.
    radius = numpy.cos(numpy.radians(strain_data[iii, 1]))*EARTH_RADIUS
    # Compute dx e dy [m]
    dx = cell_dx * (2*constants.pi*radius)/360 *1e3 # dx in m
    dy = cell_dy * (2*constants.pi*EARTH_RADIUS)/360 * 1e3 # dy in m
    area = dx*dy # Cell area in m^2
    #
    # Tectonic moment calculation. We use the formulation proposed 
    # by Bird and Kreemer (2015) equation 5 page 4
    mo = const*area
    mo[e2rate>=0] = mo[e2rate>=0]*(-2.)*e1rate[e2rate>=0]/1e9 # From nano strain -> strain
    mo[e2rate<0] = mo[e2rate<0]*2.*e3rate[e2rate<0]/1e9

    return sum(mo)

def get_moment_for_polygon(polygon, strain_data, strain_idx,
                           coupl_seism_thickness, shear_modulus,
                           cell_dx, cell_dy):
    """
    The strain model is stored in a hdf5 file and has a spatial index stored
    in a pickle file.
    
    :parameter polygon:
        An instance of :class:`~openquake.hazardlib.geo.polygon.Polygon`    
    :parameter strain_model:
        The 
    :parameter coupl_seism_thickness:
        This is the product of the seismogenic thickness [in m] and the 
        coupling coefficient
    :parameter shear_modulus:
        Shear modulus in Nm-2
    """
    #
    grd_lo = numpy.array(strain_data[:,0])
    grd_la = numpy.array(strain_data[:,1])
    #
    # Polygon 
    lons = polygon.lons
    lats = polygon.lats
    #
    # Bounding box 
    min_lon = numpy.min(lons) 
    max_lon = numpy.max(lons)     
    min_lat = numpy.min(lats) 
    max_lat = numpy.max(lats) 
    # 
    # Get grid cells within the polygon defining an area. The indexes of the 
    # cells included are in the 'iii' variable
    idxs = list(strain_idx.intersection((min_lon, min_lat, max_lon, max_lat)))
    iii = get_idx_points_inside_polygon(grd_lo[idxs], 
                                        grd_la[idxs],
                                        lons, 
                                        lats, 
                                        idxs)
    # 
    # Get moment 
    mo = _get_moment_from_nodes(iii, strain_data, strain_idx,
                                coupl_seism_thickness, shear_modulus,
                                cell_dx, cell_dy)
    
    return mo
