import re
import scipy
import pathlib
import shutil
import geojson as gj
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from openquake.baselib import sap


def find_file(path, imt_key):
    for fname in path.glob("hazard_curve-mean*.csv"):
        if imt_key.upper() in str(fname):
            return fname
    return None

def get_imls(keys):
    pattern = '^poe-([0-9]*\\.[0-9]*)'
    poes = []
    keys_imls = []
    for key in keys:
        if 'poe' in key:
            mtch = re.match(pattern, key)
            poes.append(float(mtch.group(1)))
            keys_imls.append(key)
    return np.array(poes), keys_imls

def get_spatial_index(df):
    data = np.stack([df.lon, df.lat]).T
    sidx = scipy.spatial.KDTree(data)
    return sidx, data

def fix_duplicates(root, mosaic, imt, sc):

    folder_name = pathlib.Path(f'{root}/{imt}-{sc}/')
    folder_mosaic = pathlib.Path(mosaic)
    # Read buffer
    buffer_fname = folder_name / "map_buffer.json"
    buffer_fname_cp = folder_name / "orig_map_buffer.json"
    
    # duplicate file
    shutil.copyfile(buffer_fname, buffer_fname_cp)
    models = sorted([m for m in sorted(folder_name.glob('map_*.json'))])
    df_buffer = gpd.read_file(str(buffer_fname))
    df_buffer.lon = np.array([d.x for d in df_buffer.geometry])
    df_buffer.lat = np.array([d.y for d in df_buffer.geometry])

    # get spatial index
    sidx_buf, data = get_spatial_index(df_buffer)
    
    fname_out = f'check_points_{imt}{sc}.txt'
    fou = open(fname_out, 'w')
    
    indexes = {}
    features = []

    
    for i_a in range(len(models)):
    
        # Skip the file with the buffer
        fname = models[i_a]
        if re.search('buffer', str(fname)):
            continue
    
        # Get the model key
        search = re.search('^map_([a-z]+)\\.', str(fname.name))
        key = search.group(1)
        print(str(fname), key)
    
        # TODO remove
        #if key not in ['eur', 'naf']:
        #    continue
    
        # Read geodataframe
        print('   read geojson')
        df_model = gpd.read_file(str(fname))
    
        # Create spatial index
        tmp = np.stack([df_model.lon.to_numpy(), df_model.lat.to_numpy()]).T
        sidx_model = scipy.spatial.KDTree(tmp)
    
        # Find points in common
        num = sidx_buf.count_neighbors(sidx_model, r=0.001)
        fou.write(f'{fname.name} {num}\n')
    
        # Find the points
        if num > 0:
            
            # Find the name of the file with the original results
            folder_orig = folder_mosaic / key.upper() / 'out' 
            fname_orig = find_file(folder_orig, imt)
            print('   read original csv')
            df_orig = pd.read_csv(folder_orig / fname_orig, skiprows=1)
            
            # Read the original results
            data_orig = np.stack([df_orig.lon.to_numpy(), df_orig.lat.to_numpy()]).T
            sidx_orig = scipy.spatial.KDTree(data_orig)
    
            # Find indexes of points in common
            idxs = sidx_buf.query_ball_tree(sidx_model, r=0.001)
    
            indexes[key] = idxs
            print('-->', num)
    
            sid = 0
            for i_1, lst in enumerate(idxs):
                
                if len(lst):
                    for i_2 in sorted(idxs[i_1], reverse=True):
                        
                        #import pdb; pdb.set_trace()
    
                        prop = {'model': key}
                        fea = gj.Feature(geometry=gj.Point((data[i_1, 0], data[i_1, 1])), properties = prop)
                        features.append(fea)
                
                        # Find the corresponding point in the original results
                        idxs_orig = sidx_orig.query_ball_point(data[i_1, :], r=0.001)
                        imls, keys_imls = get_imls(df_orig.columns)
    
                        # Updating site ID
                        sid += 1
    
    fcoll = gj.FeatureCollection(features)
    with open(f'duplicated_points-{imt}-{sc}.geojson', 'w') as f:
        gj.dump(fcoll, f)
      
    fou.close()
    all = []

    # get indices of poitns that should be removed
    for key in indexes:
        if len(indexes[key]):
            print(key)
            for idx, lst in enumerate(indexes[key]):
                if len(lst) > 0:
                    all.append(idx)

    # remove the duplicated points
    # Removing
    for idx in all:
        df_buffer.drop(idx, inplace=True)

    # write new buffer file
    df_buffer.to_file(f'{folder_name}/map_buffer.json', driver='GeoJSON')


#fix_duplicates(root, folder_mosaic, imt, sc)

fix_duplicates.root = 'Folder containing all hazard curves'
fix_duplicates.mosaic = 'Folder with mosaic'
fix_duplicates.imt = 'IMT of curves to remove dups'
fix_duplicates.sc = 'Vs30 or rock'

if __name__ == "__main__":
    sap.run(fix_duplicates)