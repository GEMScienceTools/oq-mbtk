
import re
import numpy
from shapely import wkt

from osgeo import ogr
from copy import deepcopy

from prettytable import PrettyTable

from openquake.hmtk.seismicity.selector import CatalogueSelector
from openquake.hmtk.sources.area_source import mtkAreaSource
from openquake.hazardlib.const import TRT

from oqmbt.tools.general import _get_point_list
from oqmbt.tools.geo import get_idx_points_inside_polygon
from oqmbt.oqt_project import OQtSource

from openquake.hazardlib.geo.polygon import Polygon


def load_geometry_from_shapefile(shapefile_filename):
    """
    :parameter str shapefile_filename:
        Name of the shapefile containing the polygons
    :parameter Boolean log:
        Flag controlling information while processing
    :returns:
        A list of :class:`oqmbt.oqt_project.OQtSource` istances
    """
    idname = 'Id'
    # Set the driver
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.Open(shapefile_filename, 0)
    layer = datasource.GetLayer()
    # Reading sources geometry
    sources = {}
    id_set = set()
    for feature in layer:
        #
        # Read the geometry
        geom = feature.GetGeometryRef()
        polygon = wkt.loads(geom.ExportToWkt())
        x, y = polygon.exterior.coords.xy
        points = _get_point_list(x, y)
        #
        # Set the ID
        if isinstance(feature.GetField(idname), str):
            id_str = feature.GetField(idname)
        elif isinstance(feature.GetField(idname), int):
            id_str = '%d' % (feature.GetField(idname))
        else:
            raise ValueError('Unsupported source ID type')
        src = OQtSource(source_id=id_str,
                        source_type='AreaSource',
                        polygon=Polygon(points),
                        name=id_str,
                        )
        # Append the new source
        if not id_set and set(id_str):
            sources[id_str] = src
        else:
            raise ValueError('Sources with non unique ID %s' % id_str)
    return sources


def src_oqt_to_hmtk(src):
    return mtkAreaSource(
            identifier=src.source_id,
            name=src.name,
            geometry=src.polygon)


def create_catalogue(model, catalogue, area_source_ids_list=None,
                     polygon=None):
    """
    Select earthquakes within the polygon delineating an area source

    :parameter model:
        An :class:`oqmbt.oqt_project.OQtModel` instance
    :parameter catalogue:
        A hmtk-formatted catalogue
    :parameter area_source_ids_list:
        The list of source ID to be used for the selection. The IDs are the
        ones of one or several sources in the `sources` attribute of the
        `model` instance.
    :parameter polygon:
        A polygon (used when the area_source_ids_list is None).
    :returns:
        Returns an hmtk-formatted catalogue containing only the earthquakes
        inside the polygon
    """
    #
    #
    #
    #
    if area_source_ids_list is not None:
        if len(area_source_ids_list) > 1:
                msg = 'We do not support the selection for multiple sources'
                raise ValueError(msg)
        # Process the area source
        src_id = area_source_ids_list[0]
        src = model.sources[src_id]
        # Set the geometry
        if 'polygon' in src.__dict__:
            # The source has a polygon
            pass
        elif src_id in model.nrml_sources:
            # Set the polygon from nrml
            src.polygon = model.nrml_sources[src_id].polygon
            src.name = model.nrml_sources[src_id].name
            src.source_id = model.nrml_sources[src_id].source_id
        else:
            print('The source does not have a geometry assigned')
            return None
    elif polygon is not None:
        assert isinstance(polygon, Polygon)
        src_id = 'user_defined'
        src = OQtSource('id', 'AreaSource')
        src.name = 'dummy'
        src.polygon = polygon
    else:
        msg = 'Either a polygon or a list of src id must be defined'
        raise ValueError(msg)
    #
    #
    neqk = len(catalogue.data['longitude'])
    sel_idx = numpy.full((neqk), False, dtype=bool)
    pnt_idxs = [i for i in range(0, neqk)]
    idxs = get_idx_points_inside_polygon(catalogue.data['longitude'],
                                         catalogue.data['latitude'],
                                         src.polygon.lons, src.polygon.lats,
                                         pnt_idxs, buff_distance=0.)
    sel_idx[idxs] = True
    #
    # Select earthquakes
    cat = deepcopy(catalogue)
    selector = CatalogueSelector(cat, create_copy=False)
    selector.select_catalogue(sel_idx)
    #
    # set label
    labels = ['%s' % src_id for i in range(0, len(cat.data['longitude']))]
    cat.data['comment'] = labels
    # Complete the composite subcatalogue
    return cat


def create_gr_table(model):
        # Set table
        p = PrettyTable(["ID", "a_gr", "b_gr"])
        p.align["Source ID"] = 'l'
        p.align["a_gr"] = 'r'
        p.align["b_gr"] = 'r'
        #
        for key in sorted(model.sources):
            src = model.sources[key]
            if src.source_type == 'AreaSource':
                alab = ''
                blab = ''
                if 'a_gr' in src.__dict__:
                    alab = '%8.5f' % (src.a_gr)
                if 'b_gr' in src.__dict__:
                    blab = '%6.3f' % (src.b_gr)
                p.add_row([key, alab, blab])
        return p


def create_mmax_table(model):
        # Set table
        p = PrettyTable(["ID", "mmax obs", "mmax assigned", "mo strain"])
        p.align["Source ID"] = 'l'
        p.align["mmax obs"] = 'r'
        p.align["mmax assigned"] = 'r'
        p.align["mo strain"] = 'r'
        #
        for key in sorted(model.sources):
            src = model.sources[key]
            if src.source_type == 'AreaSource':
                alab = ''
                blab = ''
                clab = ''
                if 'mmax_obs' in src.__dict__:
                    alab = '%6.2f' % (src.mmax_obs)
                if 'mmax_expected' in src.__dict__:
                    blab = '%6.2f' % (src.mmax_expected)
                if 'mo_strain' in src.__dict__:
                    clab = '%6.2e' % (src.mo_strain)
                p.add_row([key, alab, blab, clab])
        return p


def plot_area_source_polygons(model, bmap):
        """
        :parameter bmap:
                A :class:Basemap instance
        """
        for key in sorted(model.sources):
            src = model.sources[key]
            if src.source_type == 'AreaSource':
                        x, y = bmap(src.polygon.lons, src.polygon.lats)
                        bmap.plot(x, y, '-b')


def _set_trt(tstr):
    if re.match('ACT', tstr):
        return TRT.ACTIVE_SHALLOW_CRUST
    elif re.match('STA', tstr):
        return TRT.STABLE_CONTINENTAL
    elif re.match('DEEP', tstr):
        return 'Deep'
    else:
        raise ValueError('Unrecognized tectonic region type')


def areas_to_oqt_sources(shapefile_filename):
    """
    :parameter str shapefile_filename:
        Name of the shapefile containing the polygons
    :parameter Boolean log:
        Flag controlling information while processing
    :returns:
        A list of :class:`oqmbt.oqt_project.OQtSource` istances
    """
    idname = 'Id'
    # Set the driver
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.Open(shapefile_filename, 0)
    layer = datasource.GetLayer()
    # Reading sources geometry
    sources = {}
    id_set = set()
    for feature in layer:
        # Read the geometry
        geom = feature.GetGeometryRef()
        polygon = wkt.loads(geom.ExportToWkt())
        x, y = polygon.exterior.coords.xy
        points = _get_point_list(x, y)
        # Set the ID
        if isinstance(feature.GetField(idname), str):
            id_str = feature.GetField(idname)
        elif isinstance(feature.GetField(idname), int):
            id_str = '%d' % (feature.GetField(idname))
        else:
            raise ValueError('Unsupported source ID type')
        # Set tectonic region
        trt = _set_trt(feature.GetField('TectonicRe'))
        # Set lower seismogenic depth
        lsd = float(feature.GetField('Depth'))
        # Set coupling coefficient
        coupc = float(feature.GetField('coup_coef'))
        # Set coupling coefficient
        coupt = float(feature.GetField('coup_thick'))
        # Create the source
        src = OQtSource(source_id=id_str,
                        source_type='AreaSource',
                        polygon=Polygon(points),
                        name=id_str,
                        lower_seismogenic_depth=lsd,
                        tectonic_region_type=trt,
                        )
        src.coup_coef = coupc
        src.coup_thick = coupt
        # Append the new source
        if not id_set and set(id_str):
            sources[id_str] = src
        else:
            raise ValueError('Sources with non unique ID %s' % id_str)

    datasource.Destroy()
    return sources
