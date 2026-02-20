import os
import geopandas as gpd
import pandas as pd
import numpy as np
from openquake.baselib import sap
from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.sourceconverter import SourceConverter
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.scalerel import PointMSR
from openquake.hazardlib.sourcewriter import write_source_model
from openquake.wkf.distributed_seismicity import _get_point_sources, from_list_ps_to_multipoint


def crop_mps_to_poly (src_xml_fname, fname_poly,  out_folder, keep_out, src_conv=False, poly_id = []):
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
    if src_conv:
        tssm = to_python(src_xml_fname, src_conv)
    else:
        tssm = to_python(src_xml_fname)

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
        poly_id =  poly_proj.id.iloc[i]
        print('making source ', poly_id)
    
        # Find index of the sources within the polygon
        poly_df = gpd.GeoDataFrame(poly_proj[poly_proj.id == poly_id])
        idx_within = poly_df.sindex.query(mesh.geometry, predicate="intersects")[0]
        idx_out = [i for i in range(len(mesh.geometry)) if not i in idx_within]  
        
        keep_outside = bool(keep_out)
        if keep_outside:
            idx_keep = idx_out
        else:
            idx_keep = idx_within

        ip2_idx_list = []
        for idx in range(0, len(idx_keep)):
            pt = pnt_srcs[idx_keep[idx]]
            ip2_idx_list.append(pt)
    

        # Create the multi-point source
        tmpsrc = from_list_ps_to_multipoint(ip2_idx_list, 'pnts_{}'.format(poly_id))
        # Make out_name from id in input
        fname_out = os.path.join(out_folder, f'{src_xml_fname.split('/')[-1].replace('.xml','')}_{poly_id}.xml')
        #fname_out = os.path.join(out_folder, 'src_{}.xml'.format(poly_id))
        print('saving source to ', fname_out)

        write_source_model(fname_out, [tmpsrc], 'Distributed seismicity {}'.format(poly_id))

#one day we might add so that the src_conv can be a command line or toml object
#but for now it's not possible to run from command line 

def crop_mps (src_xml_fname, fname_poly, out_folder, keep_outside=0, src_conv=None, poly_id = []):
    """
    crop multipoint source to given polygon
    source will retain original mmax, nodal plane distribution etc
    """
    crop_mps_to_poly (src_xml_fname, fname_poly, out_folder, keep_outside, src_conv, poly_id)

crop_mps.src_xml = 'Source xml file to be cropped'
crop_mps.fname_poly = 'file name of source polygon'
crop_mps.out_folder = 'output folder to store new (cropped) mps'
crop_mps.keep_out = '1 if keeping points outside polygon else 0'
crop_mps.src_conv = 'source converter object'
crop_mps.poly_id = 'optional list of polygon ids to crop'

if __name__=="__main__":
    sap.run(crop_mps)
