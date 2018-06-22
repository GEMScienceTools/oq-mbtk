
import math
import numpy

from openquake.hazardlib.geo.mesh import Mesh


def get_fault_vertices_3d(fault_trace, upper_seismogenic_depth,
                          lower_seismogenic_depth, dip):
    """
    Get surface main vertexes.

    Parameters are the same as for :meth:`from_fault_data`, excluding
    mesh spacing.

    :returns:
        Coordinates of fault surface vertexes in Longitude, Latitude, and
        Depth.
        The order of vertexs is given clockwisely
    """
    # Similar to :meth:`from_fault_data`, we just don't resample edges
    dip_tan = math.tan(math.radians(dip))
    hdist_bottom = lower_seismogenic_depth / dip_tan

    strike = fault_trace[0].azimuth(fault_trace[-1])
    azimuth = (strike + 90.0) % 360

    # Collect coordinates of vertices on the top and bottom edge
    lons = []
    lats = []
    deps = []

    t_lon = []
    t_lat = []
    t_dep = []

    for point in fault_trace.points:
        top_edge_point = point.point_at(0, 0, 0)
        bottom_edge_point = point.point_at(hdist_bottom, 0, azimuth)

        lons.append(top_edge_point.longitude)
        lats.append(top_edge_point.latitude)
        deps.append(upper_seismogenic_depth)
        t_lon.append(bottom_edge_point.longitude)
        t_lat.append(bottom_edge_point.latitude)
        t_dep.append(lower_seismogenic_depth)

    all_lons = numpy.array(lons + list(reversed(t_lon)), float)
    all_lats = numpy.array(lats + list(reversed(t_lat)), float)
    all_deps = numpy.array(deps + list(reversed(t_dep)), float)

    return all_lons, all_lats, all_deps


def _get_rate_above_m_low(seismic_moment, m_low, m_upp, b_gr):
    """
    :parameter seismic_moment:
        Seismic moment in Nm
    :parameter m_low:
        Lower magnitude threshold
    :parameter m_upp:
        Upper magnitude threshold
    :parameter b_gr:
        b value of the Gutenberg-Richter relationship
    """
    a_m = 9.1
    b_m = 1.5
    beta = b_gr * numpy.log(10.)
    x = (-seismic_moment*(b_m*numpy.log(10.) - beta) /
         (beta*(10**(a_m + b_m*m_low) -
          10**(a_m + b_m*m_upp)*numpy.exp(beta*(m_low - m_upp)))))
    rate_m_low = x * (1-numpy.exp(-beta*(m_upp-m_low)))
    return rate_m_low


def _get_cumul_rate_truncated(m, m_low, m_upp, rate_gt_m_low, b_gr):
    """
    This is basically equation 9 of Youngs and Coppersmith (1985)
    """
    beta = b_gr * numpy.log(10.)
    nmr1 = numpy.exp(-beta*(m-m_low))
    nmr2 = numpy.exp(-beta*(m_upp-m_low))
    den1 = 1-numpy.exp(-beta*(m_upp-m_low))
    rate = rate_gt_m_low * (nmr1 - nmr2) / den1
    return rate


def rates_for_double_truncated_mfd(area, slip_rate, m_low, m_upp, b_gr,
                                   bin_width=0.1, rigidity=32e9):
    """
    :parameter area:
        Area of the fault surface
        float [km2]
    :parameter slip_rate:
        Slip-rate
        float [mm/tr]
    :parameter m_low:
        Lower magnitude
        float
    :parameter m_upp:
        Upper magnitude
    :parameter b_gr:
        b-value of Gutenber-Richter relationship
    :parameter bin_width:
        Bin width
    :parameter rigidity:
        Rigidity [Pa]
    :return:
        A list containing the rates per bin starting from m_low
    """
    #
    # Compute moment
    slip_m = slip_rate * 1e-3  # [mm/yr] -> [m/yr]
    area_m2 = area * 1e6
    moment_from_slip = (rigidity * area_m2 * slip_m)
    #
    # Compute total rate
    rate_above = _get_rate_above_m_low(moment_from_slip, m_low, m_upp, b_gr)
    #
    # Compute rate per bin
    rrr = []
    mma = []
    for mmm in numpy.arange(m_low, m_upp, bin_width):
        rte = (_get_cumul_rate_truncated(mmm, m_low, m_upp, rate_above, b_gr) -
               _get_cumul_rate_truncated(mmm+bin_width, m_low,
                                         m_upp, rate_above, b_gr))
        mma.append(mmm+bin_width/2)
        rrr.append(rte)
    return rrr


def get_points_within_distance(surface, points):
    """
    :parameters surface
        An instance of
        :class:`openquake.hazardlib.geo.surface.SimpleFaultSurface`
    :parameters list points
        A list of :class:`openquake.hazardlib.geo.point.Point` instances
    """

    mesh = Mesh.from_points_list(points)
    dst = surface.mesh.get_min_distance(mesh)
    return dst


def get_surface_corners_coordinates(surface):
    """
    :parameters surface
        An instance of
        :class:`openquake.hazardlib.geo.surface.SimpleFaultSurface`
    """
    mesh = surface.mesh
