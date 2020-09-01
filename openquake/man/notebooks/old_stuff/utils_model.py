import pyproj
import shapely.ops as ops
from functools import partial
from openquake.hazardlib.sourceconverter import SourceConverter
from openquake.hazardlib.nrml import SourceModelParser


def get_source_model(source_file, inv_time, simple_mesh_spacing=1.0,
                     complex_mesh_spacing=10.0, mfd_spacing=0.1,
                     area_discretisation=10.):
    """
    Read and build a source model from an xml file
    """
    conv = SourceConverter(inv_time, simple_mesh_spacing, complex_mesh_spacing,
                           mfd_spacing, area_discretisation)
    parser = SourceModelParser(conv)
    return parser.parse_src_groups(source_file)


def read_model(model_filename):
    """
    This reads the nrml file containing the model

    :parameter model_filename:
    :return:
        A list of sources
    """
    # Analysis
    print('Reading: %s' % (model_filename))
    source_grps = get_source_model(model_filename, inv_time=1)
    source_model = []
    for grp in source_grps:
        for src in grp.sources:
            source_model.append(src)
    print('The model contains %d sources' % (len(source_model)))
    return source_model


def get_area(geom):
    """
    Compute the area of a shapely polygon with lon, lat coordinates.
    See http://tinyurl.com/h35nde4

    :parameter geom:
    :return:
        The area of the polygon in km2
    """
    geom_aea = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat_1=geom.bounds[1],
                lat_2=geom.bounds[3])),
        geom)
    return geom_aea.area/1e6
