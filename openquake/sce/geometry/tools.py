

import numpy
from openquake.hazardlib.geo.line import Line
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.geodetic import point_at
from openquake.hazardlib.geo.surface.simple_fault import SimpleFaultSurface


def tor2trace(tor_trace, dip, tor_depth=None):
    """
    :param tor_trace:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    :param dip:
        A positive real defining the average dip angle
    :param tor_depth:
        A positive real defining the depth to the top-of-rupture [km]
    :return:
        An xml string
    """

    azimuth = tor_trace.average_azimuth()
    if tor_depth is None:
        depths = numpy.array([p.depth for p in tor_trace])
        assert numpy.all(depths-depths[0] < 1e-6)
        tor_depth = depths[0]
    delta = tor_depth * numpy.tan(numpy.radians(dip))
    trace_points = []
    translate_azimuth = (azimuth - 90.) % 360
    for p in tor_trace:
        lon, lat = point_at(p.longitude, p.latitude, translate_azimuth, delta)
        trace_points.append(Point(lon, lat, 0.0))
    return Line(trace_points)


def get_sf_hypocenter(fault_trace, upp_sd, low_sd, dip, mesh_spacing):
    """
    See
    """
    sfc = SimpleFaultSurface.from_fault_data(fault_trace, upp_sd, low_sd, dip,
                                             mesh_spacing)
    shape = sfc.mesh.shape
    idx_row = int(numpy.floor(shape[0]/2))
    idx_col = int(numpy.floor(shape[1]/2))
    print(type(sfc.mesh.lons))
    hypo = Point(sfc.mesh.lons[idx_row, idx_col],
                 sfc.mesh.lats[idx_row, idx_col],
                 sfc.mesh.depths[idx_row, idx_col])
    return hypo


def get_sf_geometry_xml(trace, upp_sd, low_sd, mag, rake, dip):
    """
    :param trace:
        An instance of :class:`openquake.hazardlib.geo.line.Line`
    :param upp_sd:
        A positive real defining the top depth of the rupture
    :param low_sd:
        A positive real defining the bottom depth of the rupture
    :param mag:
        A real defining the magnitiude of the rupture. Mw is assumed.
    :param rake:
        A real in [-90, 90] defining the slip direction. Obeys Aki & Richards
        convention
    :param dip:
        A positive real defining the average dip angle
    """
    space = '    '
    xml = '{:s}<simpleFaultRupture>\n'.format(space)
    xml += '{:s}<magnitude>{:.2f}</magnitude>\n'.format(2*space, mag)
    xml += '{:s}<rake>{:.2f}</rake>\n'.format(2*space, rake)
    hypo = get_sf_hypocenter(trace, upp_sd, low_sd, dip, 5.0)
    fmt = '{:s}<hypocenter lat=\"{:f}\" lon=\"{:f}\" depth=\"{:f}\" />\n'
    xml += fmt.format(2*space, hypo.longitude, hypo.latitude, hypo.depth)
    xml += '{:s}<simpleFaultGeometry>\n'.format(2*space)
    xml += '{:s}<gml:LineString>\n'.format(3*space)
    xml += '{:s}<gml:posList>\n'.format(4*space)
    for pnt in trace:
        xml += '{:s}{:f} {:f}\n'.format(5*space, pnt.longitude, pnt.latitude)
    xml += '{:s}</gml:posList>\n'.format(4*space)
    xml += '{:s}</gml:LineString>\n'.format(3*space)
    xml += '{:s}<dip>{:.2f}</dip>\n'.format(3*space, dip)
    fmt = '{:s}<upperSeismoDepth>{:.2f}</upperSeismoDepth>\n'
    xml += fmt.format(3*space, upp_sd)
    fmt = '{:s}<lowerSeismoDepth>{:.2f}</lowerSeismoDepth>\n'
    xml += fmt.format(3*space, low_sd)
    xml += '{:s}</simpleFaultGeometry>\n'.format(2*space)
    xml += '{:s}</simpleFaultRupture>'.format(space)
    return xml
