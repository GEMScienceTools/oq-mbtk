import numpy as np
import pandas as pd
import matplotlib.path as mpath

from openquake.hazardlib.geo import Point, Line
from openquake.hazardlib.geo.surface import KiteSurface

from openquake.fnm.rupture_connections import get_all_contiguous_subfaults

def points_in_triangles(points, triangles):
    """
    Determines which triangle from a set each point belongs to.
    
    Parameters:
    - points: An array of points with shape (N,2).
    - triangles: An array of triangles with shape (M,3,2), where each triangle 
      is represented by three vertices.
    
    Returns:
    - A numpy array of indices with the same length as points, where each
      element is the index of the triangle that the point belongs to, or -1 if 
      the point is not in any triangle.
    """
    associations = np.full(len(points), -1, dtype=int)  # Initialize with -1
    
    for i, triangle in enumerate(triangles):
        # Create a path for the current triangle
        path = mpath.Path(triangle)
        
        # Find points contained in this triangle
        contained = path.contains_points(points)
        
        # For points contained in the triangle, update their association
        associations[contained] = i
    
    return associations


def preprocess_triangles(geojson_features):
    """
    Preprocess triangles from GeoJSON features, extracting paths and properties.
    
    Parameters:
    - geojson_features: A list of GeoJSON features representing triangles.
    
    Returns:
    - A tuple of (paths, properties) where paths is a list of Matplotlib paths
      for each triangle,
      and properties is a list of dictionaries containing properties for each
      triangle.
    """
    paths = []
    properties = []

    print(f"Processing tris")
    for i, feature in enumerate(geojson_features):
        print(f"\tdoing tri {str(i+1).zfill(4)} / {len(geojson_features)}",
              end="\r")
        
        # Extract the coordinates for the triangle vertices
        vertices = np.array(feature['geometry']['coordinates'][0])
        # Create a Matplotlib path for the triangle
        path = mpath.Path(vertices[:-1,0:2])  # Remove duplicate vertex
        paths.append(path)
        
        # Store the properties
        properties.append(feature['properties'])
    print("")
    
    return paths, properties


def analyze_points_in_triangles(points_arrays, paths, properties, 
                                property_keys):
    """
    For each array of points, find which triangle contains each point and 
    calculate mean properties.
    
    Parameters:
    - points_arrays: A list of arrays, each containing points to test against 
      the triangles.
    - paths: A list of Matplotlib paths representing the triangles.
    - properties: A list of property dictionaries for each triangle.
    - property_keys: The keys of the properties for which to calculate means.
    
    Returns:
    - A dict keyed by the index of the points array containing the results.
    """
    results = {}

    print(f"processing point sets")

    for i, points in enumerate(points_arrays):
        print(f"\tdoing point set {str(i+1).zfill(4)} / {len(points_arrays)}", 
              end="\r")
        
            
        triangle_indices = np.full(len(points), -1, dtype=int)
        points_properties = {key: [] for key in property_keys}
        
        for j, path in enumerate(paths):
            contained = path.contains_points(points)
            triangle_indices[contained] = j
            for key in property_keys:
                points_properties[key].extend(
                    [properties[j][key]] * np.sum(contained))
        
        # Calculate mean properties for the points in this array
        mean_properties = {}
        for k, v in points_properties.items():
            if v:
                if k == 'fid':
                    mean_properties[k]= v[0]
                else:
                    mean_properties[k] = np.mean(v)
        
        results[i] = {
            'triangle_indices': triangle_indices,
            #'properties': points_properties
            'properties': mean_properties
        }
    print("")
    
    return results


def sub_mesh_pts(sub):
    lons = sub['mesh'].lons.ravel()
    lats = sub['mesh'].lats.ravel()
    depths=sub['mesh'].depths.ravel()

    return np.vstack((lons, lats)).T


def sub_to_subfault(sub, sub_prop, i, fid_base='interface'):
    subfault = {
        'fid': fid_base + f"_{str(i).zfill(4)}",
        'net_slip_rate': sub_prop['properties']['net_slip_rate'],
        'net_slip_rate_err': sub_prop['properties']['net_slip_rate_err'],
        'rake': sub_prop['properties']['rake'],
        'fault_position': (sub['row'], sub['col']),
        'surface': KiteSurface(sub['mesh']),
        'trace': [
            [lon, sub['mesh'].lats[0, i], sub['mesh'].depths[0, i]]
            for i, lon in enumerate(sub['mesh'].lons[0])
        ],
        }
    subfault['length'] = Line(
            [Point(*p) for p in subfault['trace']]
    ).get_length()
    subfault['width'] = subfault['surface'].get_width()
    subfault["area"] = subfault["surface"].get_area()
    subfault["strike"] = subfault["surface"].get_strike()
    subfault["dip"] = subfault["surface"].get_dip()
    subfault["subsec_id"] = i

    return subfault


def get_rupture_patches_from_kite_fault(
    subfaults,
    min_aspect_ratio: float = 0.8,
    max_aspect_ratio: float = 3.0,
    identifier='id',
) -> dict:
    """
    Get all possible contiguous subfaults from a single fault, within
    the specified aspect ratio bounds.

    Parameters
    ----------
    subfaults : list of dictionaries
        List of subfault dictionaries.
    min_aspect_ratio : float, optional
        Minimum aspect ratio of the rupture. The default is 0.8.
    max_aspect_ratio : float, optional
        Maximum aspect ratio of the rupture. The default is 3.0.

    Returns
    -------
    dict
        Dictionary of ruptures. The keys are the fault identifiers, and the
        values are lists of lists of subfault indices.
    """
    num_rows = len(np.unique([sf['fault_position'][0] for sf in subfaults]))
    num_cols = len(np.unique([sf['fault_position'][1] for sf in subfaults]))

    subfault_quick_lookup = {
        sf['fault_position']: i for i, sf in enumerate(subfaults)
    }

    identifier = subfaults[0]['fid']
    sub_length = subfaults[0]['length']
    sub_width = subfaults[0]['width']

    single_fault_rup_indices = get_all_contiguous_subfaults(
        num_cols,
        num_rows,
        s_length=sub_length,
        d_length=sub_width,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )

    single_fault_rups = []
    for rup in single_fault_rup_indices:
        try:
            rr = [subfault_quick_lookup[pos] for pos in rup]
            single_fault_rups.append(rr)
        except KeyError:
            pass


    return {identifier: single_fault_rups}


def get_single_fault_rups_kite(
    subfaults, subfault_index_start: int = 0, identifier='id',
) -> pd.DataFrame:
    """
    Get all possible ruptures from a single fault.

    Parameters
    ----------
    subfaults : list of dictionaries
        List of subfault dictionaries.
    subfault_index_start : int, optional
        Index of the first subfault. The default is 0.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the rupture information. The columns are:
            fault_rup: rupture index
            patches: list of patch indices in the rupture
            subfaults: list of subfault indices in the rupture
            fault: fault identifier
    """
    num_subfaults = len(subfaults)
    fault_rups = get_rupture_patches_from_kite_fault(subfaults, identifier=identifier)
    rup_patches = list(fault_rups.values())[0]
    rup_subfaults = [
        [rp + subfault_index_start for rp in rup] for rup in rup_patches
    ]
    num_rups = len(rup_patches)

    rupture_df = pd.DataFrame(
        index=np.arange(num_rups) + subfault_index_start,
        data={
            'fault_rup': np.arange(num_rups),
            'patches': rup_patches,
            'subfaults': rup_subfaults,
        },
    )
    rupture_df['fault'] = list(fault_rups.keys())[0]
    rupture_df['full_fault_rupture'] = [
        len(rup) == num_subfaults for rup in rup_patches
    ]

    return rupture_df
