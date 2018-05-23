# coding: utf-8

import os
import sys
import re

from shapely import wkt
from osgeo import ogr

from openquake.mbt.tools.model import read

from openquake.hazardlib.source import SimpleFaultSource
from openquake.hazardlib.geo.line import Line

from openquake.mbt.oqt_project import OQtProject, OQtSource
from openquake.mbt.tools.mfd import get_moment_from_mfd

from openquake.mbt.tools.utils import _get_point_list


MAPPING = {'identifier': 'ID',
           'name': 'NAME',
           'dip': 'DIP',
           'rake': 'RAKE',
           'upper_depth': 'DMIN',
           'lower_depth': 'DMAX',
           'recurrence': 'RECURRENCE',
           'slip_rate': 'Sliprate',
           'aseismic': 'COEF',
           'mmax': 'MMAX',
           'ri': 'RECURRENCE',
           'coeff_fault': 'COEF',
           'use_slip': 'SLIPSELECT',
           }


def shallow_faults_to_oqt_sources(shapefile_filename, mapping=None):
    """
    Creates al list of mtkActiveFault istances starting from a shapefile

    :parameter string shapefile_filename:
        Name of the shapefile containing information on active faults
    :parameter mapping:
        A dictionary indicating for each parameter in the shapefile attribute
        table the corresponding one in the mtkActiveFault object. Note that
        only the information listed in this dictionary will be included in the
        mtkActiveFault istances.
    """
    # set the default mapping. This is based on the information in the
    # attribute table of the shapefile Yufang sent in Sept. 2015.
    if mapping is None:
        mapping = MAPPING
    # check if file exists
    assert os.path.exists(shapefile_filename)
    # reading the file
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(shapefile_filename, 0)
    layer = dataSource.GetLayer()
    # reading sources geometry
    src_id = set()
    sources = {}
    for cnt, feature in enumerate(layer):
        # get dip
        dip = float(feature.GetField(mapping['dip']))
        # get geometry
        geom = feature.GetGeometryRef()
        geom_wkt = geom.ExportToWkt()

        tmp = feature.GetField(mapping['identifier'])
        if isinstance(tmp, str):
            sid = 'sf'+tmp
        elif isinstance(tmp, int):
            sid = 'sf%d' % tmp
        elif isinstance(tmp, float):
            sid = 'sf%d' % int(tmp)
        else:
            print('value ', tmp)
            print('type  ', type(tmp))
            raise ValueError('Unsupported ID type')

        if (re.search('^MULTILINESTRING', geom_wkt) or dip < 0.1 or
                feature.GetField("TYPE") == 'SUB'):
            print('skipping - multiline - src id: %s ' % (sid))
            print('   ', feature.GetField(mapping['name']))
        else:
            line = wkt.loads(geom.ExportToWkt())
            x, y = line.coords.xy

            if dip > 90 or dip < 0:
                print('dip outside admitted range')
                print('   ', feature.GetField(mapping['dip']))
                print('   ', feature.GetField(mapping['name']))

            if src_id & set(sid):
                raise ValueError('Source ID already used %s' % sid)

            src = OQtSource(sid, 'SimpleFaultSource')
            src.trace = Line(_get_point_list(x, y))
            src.dip = float(feature.GetField(mapping['dip']))
            src.upper_seismogenic_depth = (
                float(feature.GetField(mapping['upper_depth'])))
            src.lower_seismogenic_depth = float(
                feature.GetField(mapping['lower_depth']))
            src.name = feature.GetField(mapping['name'])
            src.slip_rate = float(feature.GetField(mapping['slip_rate']))
            src.recurrence = float(feature.GetField(mapping['recurrence']))
            src.rake = float(feature.GetField(mapping['rake']))
            src.mmax = float(feature.GetField(mapping['mmax']))
            src.ri = float(feature.GetField(mapping['ri']))
            src.coeff = float(feature.GetField(mapping['coeff_fault']))
            src.use_slip = float(feature.GetField(mapping['use_slip']))
            sources[sid] = src
    dataSource.Destroy()

    return sources


def read_faults(faults_xml_filename=None):
    """
    Reads the information on faults included in an .xml file

    :parameter faults_xml_filename:
        The name of the .xml file with the faults
    """
    #
    # loading project
    project_pickle_filename = os.environ.get('OQMBT_PROJECT')
    oqtkp = OQtProject.load_from_file(project_pickle_filename)
    model_id = oqtkp.active_model_id
    model = oqtkp.models[model_id]
    if faults_xml_filename is None:
        fname = getattr(model, 'faults_xml_filename')
        faults_xml_filename = os.path.join(oqtkp.directory, fname)
    #
    # read .xml file content
    sources, _ = read(faults_xml_filename)
    #
    # save the information
    for f in sources:
        #
        # fixing the id of the fault source
        sid = str(f.source_id)
        if not re.search('^fs_', sid):
            sid = 'fs_{:s}'.format(sid)
        if isinstance(f, SimpleFaultSource):
            src = OQtSource(sid, 'SimpleFaultSource')
            src.trace = f.fault_trace
            src.msr = f.magnitude_scaling_relationship
            src.mfd = f.mfd
            src.rupture_aspect_ratio = f.rupture_aspect_ratio
            src.trt = f.tectonic_region_type
            src.dip = f.dip
            src.upper_seismogenic_depth = f.upper_seismogenic_depth
            src.lower_seismogenic_depth = f.lower_seismogenic_depth
            src.name = f.name
            src.rake = f.rake
            model.sources[sid] = src
        else:
            raise ValueError('Unsupported fault type')
    #
    # save the project
    oqtkp.models[model_id] = model
    oqtkp.save()


