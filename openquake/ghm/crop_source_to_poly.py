import os
import geopandas as gpd
import pandas as pd
import numpy as np
from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.sourceconverter import SourceConverter
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.scalerel import PointMSR
from openquake.hazardlib.sourcewriter import write_source_model
from openquake.wkf.distributed_seismicity import _get_point_sources, from_list_ps_to_multipoint


def crop_mps_to_poly (src_xml_fname, fname_poly, src_conv, out_folder, poly_id = []):
    '''
    Function to crop multipoint sources down to specified polygons by extracting all points within the polygon
    and making a new source. 
        
    To be used with caution! 
    	- Will not flag any uncovered parts of polygon. 
    	- Currently tested with geojson polys
    	- will inherit TR (and therefore GMM) from original source

    :param src_xml_fname:
        path to xml source to extract points from
    :param fname_poly:
        file containing polygon(s). Function will loop through all polygons unless
        'poly_id' is specified. Sources will be named with 'id' column.
    :param src_conv:
        SourceConverter object. Should be consistent with source.
    :param out_folder:
        Output folder where new sources will be stored
    :param poly_id:
        optional: id of polygon to be used.
    
    '''
    # Load poly 
    gdf_geoms = gpd.read_file(fname_poly)
    poly_proj = gdf_geoms.to_crs('EPSG:4326')
    if poly_id:
        poly_proj = poly_proj[poly_proj.id == poly_id]

    # Get the point sources used to model distributed seismicity
    tssm = to_python(src_xml_fname, src_conv)
    wsrc = _get_point_sources(tssm)

    # List point sources 
    pnt_srcs = []
    coo_pnt_src = []
    tcoo = np.array(
          [(p.location.longitude, p.location.latitude) for p in wsrc])
    pnt_srcs.extend(wsrc)
    coo_pnt_src.extend(tcoo)
    coo_pnt_src = np.array(coo_pnt_src)

    # make a mesh using point source lon/lats
    tmp = pd.DataFrame(data={'lon': coo_pnt_src[:,0], 'lat': coo_pnt_src[:,1]})
    mesh = gpd.GeoDataFrame(tmp, geometry=gpd.points_from_xy(tmp.lon, tmp.lat), crs="EPSG:4326")

    for i in range(0, len(poly_proj.id)):
        poly_id = poly_proj.id[i]
        print('making source ', poly_id)
    
        # Find index of the sources within the polygon
        poly_df = gpd.GeoDataFrame(poly_proj[poly_proj.id == poly_id])
        idx_within = poly_df.sindex.query(mesh.geometry, predicate="intersects")[0]

        ip2_idx_list = []
        for idx in range(0, len(idx_within)):
            pt = pnt_srcs[idx_within[idx]]
            ip2_idx_list.append(pt)
    

        # Create the multi-point source
        tmpsrc = from_list_ps_to_multipoint(ip2_idx_list, 'pnts_{}'.format(poly_id))
        # Make out_name from id in input
        fname_out = os.path.join(out_folder, 'src_{}.xml'.format(poly_id))
        print('saving source to ', fname_out)

        write_source_model(fname_out, [tmpsrc], 'Distributed seismicity {}'.format(poly_id))
