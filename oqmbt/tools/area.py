
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
    dataSource = driver.Open(shapefile_filename, 0)
    layer = dataSource.GetLayer()
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
    Note that this assumes that the catalogue has a rtree spatial index
    associated.

    :parameter model:
    :parameter catalogue:
    :parameter area_source_ids_list:
    """

    if area_source_ids_list is not None:
        # Process the area source
        src_id = area_source_ids_list[0]
        src = model.sources[src_id]
        # Check if the area source has a geometry
        if 'polygon' in src.__dict__:
            pass
        elif src_id in model.nrml_sources:
            src.polygon = model.nrml_sources[src_id].polygon
            src.name = model.nrml_sources[src_id].name
            src.source_id = model.nrml_sources[src_id].source_id
        else:
            print ('The source does not have a geometry assigned')
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

    # This sets the limits of the area covered by the polygon
    limits = [numpy.min(src.polygon.lons),
              numpy.min(src.polygon.lats),
              numpy.max(src.polygon.lons),
              numpy.max(src.polygon.lats)]
    # Src hmtk
    src_hmtk = src_oqt_to_hmtk(src)
    # This creates a new catalogue with eqks within the bounding box of
    # the analysed area source
    selectorB = CatalogueSelector(catalogue, create_copy=True)
    tmpcat = selectorB.within_bounding_box(limits)
    selectorA = CatalogueSelector(tmpcat, create_copy=False)
    # This filters out the eqks outside the area source
    src_hmtk.select_catalogue(selectorA)
    # Create the composite catalogue as a copy of the sub-catalogue for the first source
    labels = ['%s' % src_id for i in range(0, len(src_hmtk.catalogue.data['magnitude']))]
    src_hmtk.catalogue.data['comment'] = labels
    fcatal = deepcopy(src_hmtk.catalogue)
    print ('merging eqks for source:', src_id, '# eqks:', len(src_hmtk.catalogue.data['magnitude']))
    # Complete the composite subcatalogue
    """
    for src_id in area_source_ids_list[1:]:
        # Set the source index and fix the catalogue selector
        src = model.sources[src_id]
        src_hmtk = src_oqt_to_hmtk(src)
        # Merge the source-subcatalogue to the composite one
        # print 'merging eqks for source:', src_id, '# eqks:', len(src_hmtk.catalogue.data['magnitude'])
        labels = ['%s' % src_id for i in range(0, len(src_hmtk.catalogue.data['magnitude']))]
        src_hmtk.catalogue.data['comment'] = labels
        fcatal.concatenate(src.catalogue)
    """
    print ('Total number of earthquakes selected {:d}'.format(
                fcatal.get_number_events()))
    return fcatal


def create_gr_table(model):
        # Set table
        p = PrettyTable(["ID","a_gr", "b_gr"])
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
        p = PrettyTable(["ID","mmax obs", "mmax assigned", "mo strain"])
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
                if src.__dict__.has_key('mmax_obs'):
                    alab = '%6.2f' % (src.mmax_obs)
                if src.__dict__.has_key('mmax_expected'):
                    blab = '%6.2f' % (src.mmax_expected)
                if src.__dict__.has_key('mo_strain'):
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
    dataSource = driver.Open(shapefile_filename, 0)
    layer = dataSource.GetLayer()
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

        """
        sources.append(mtkAreaSource(identifier=id_str,
                                     name=id_str,
                                     geometry=Polygon(points)))
        seism_thickness = float(feature.GetField('Depth'))
        coef = float(feature.GetField('coef'))
        tect_reg = feature.GetField('TectonicRe')
        sources_data[id_str] = dict(seism_thickness=float(seism_thickness),
                                    coef=coef,
                                    tect_reg=tect_reg)
        """
    dataSource.Destroy()
    return sources
