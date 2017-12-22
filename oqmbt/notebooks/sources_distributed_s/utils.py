import scipy
from oqmbt.oqt_project import OQtProject, OQtSource
from openquake.hazardlib.source import PointSource, SimpleFaultSource
from openquake.hazardlib.geo.geodetic import azimuth, point_at

def get_xy(line):
    """
    """
    x = []
    y = []
    for pnt in line.points:
        x.append(pnt.longitude)
        y.append(pnt.latitude)
    return x, y 

def get_polygon_from_simple_fault(flt):
    """
    """
    xtrace = []
    ytrace = []
    
    if isinstance(flt, SimpleFaultSource):
        trc = flt.fault_trace
    elif isinstance(flt, OQtSource):
        trc = flt.trace
        
    for pnt in trc:
        xtrace.append(pnt.longitude)
        ytrace.append(pnt.latitude)
    #
    # Get strike direction
    azim = azimuth(xtrace[0], ytrace[0],
                   xtrace[-1], ytrace[-1])
    #
    # Compute the dip direction
    dip = flt.dip
    dip_dir = (azim + 90) % 360
    seism_thickness = flt.lower_seismogenic_depth - flt.upper_seismogenic_depth 
    #
    # Horizontal distance
    h_dist = seism_thickness / scipy.tan(scipy.radians(dip))
    #
    # Compute the bottom trace
    xb = xtrace
    yb = ytrace
    for x, y in zip (xtrace[::-1], ytrace[::-1]):
        nx, ny = point_at(x, y, dip_dir, h_dist)
        xb.append(nx)
        yb.append(ny)
    
    # Create the polygon geometry
    pnt_list = []
    for x, y in zip(xb, yb):
        pnt_list.append((x,y))
    return pnt_list
