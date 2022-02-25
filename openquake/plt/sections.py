

import os
import re
from pathlib import Path
from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.sourceconverter import SourceConverter

from geojson import FeatureCollection, Feature, dump
from geojson import LineString, Polygon


def create_geojson_file(ifname: str, opath: str):
    """
    :param ifname:
        Input filename with the geometry of the sections
    :param opath:
        Output folder
    """

    # Create geometries
    conv = SourceConverter(investigation_time=1., rupture_mesh_spacing=5.)
    geomodel = to_python(ifname, conv)

    # Create the directory
    Path(opath).mkdir(parents=True, exist_ok=True)

    tors = []
    polygons = []

    for key in geomodel.sections:
        sec = geomodel.sections[key]

        tor = sec.surface.get_tor()
        coo = [[o, a] for o, a in zip(tor[0], tor[1])]
        geom = LineString(coo)
        prop = {"id": key}
        fea = Feature(geometry=geom, properties=prop)
        tors.append(fea)

        coo = []
        idxs = sec.surface._get_external_boundary_indexes()
        for i in idxs:
            coo.append([sec.surface.mesh.lons[i[0], i[1]],
                        sec.surface.mesh.lats[i[0], i[1]],
                        sec.surface.mesh.depths[i[0], i[1]]])

        # los, las = sec.surface.surface_projection
        # coo = [[o, a] for o, a in zip(los, las)]
        geom = Polygon([coo])
        prop = {"id": key}
        fea = Feature(geometry=geom, properties=prop)
        polygons.append(fea)

    feature_co = FeatureCollection(tors)
    tmp_fname = os.path.join(opath, 'tors.geojson')
    if len(tors) > 0:
        with open(tmp_fname, 'w') as fout:
            dump(feature_co, fout)
        print(f'Created gejson file: {tmp_fname}')

    feature_co = FeatureCollection(polygons)
    tmp_fname = os.path.join(opath, 'polygons.geojson')
    if len(polygons) > 0:
        with open(tmp_fname, 'w') as fout:
            dump(feature_co, fout)
        print(f'Created gejson file: {tmp_fname}')


def create_gmt_file(ifname: str, ofname: str):
    """
    :param ifname:
    :param ofname:
    """

    conv = SourceConverter(investigation_time=1., rupture_mesh_spacing=5.)
    geom = to_python(ifname, conv)

    opat, ofle = os.path.split(ofname)
    tmp = re.split('\\.', ofle)
    fnametor = os.path.join(opat, tmp[0]+'_tor.txt')
    fnamepol = os.path.join(opat, tmp[0]+'_polygon.txt')

    # Create the directory
    Path(opat).mkdir(parents=True, exist_ok=True)

    # Open files
    ftor = open(fnametor, 'w')
    fpol = open(fnamepol, 'w')
    ftor.write('>\n')
    fpol.write('>\n')

    for key in geom.sections:
        sec = geom.sections[key]

        tor = sec.surface.get_tor()
        for lo, la in zip(tor[0], tor[1]):
            ftor.write('{:.4f} {:.4f}\n'.format(lo, la))

        los, las = sec.surface.surface_projection
        for lo, la in zip(los, las):
            fpol.write('{:.4f} {:.4f}\n'.format(lo, la))

        ftor.write('>\n')
        fpol.write('>\n')

    ftor.close()
    fpol.close()
