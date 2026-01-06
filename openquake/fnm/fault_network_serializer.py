"""
Serializer and deserializer for fault rupture data structures.
Version 2: Optimized for speed.

Strategy:
- Use JSON for structure and simple types (fast, single write)
- Use HDF5 datasets only for large numerical data (arrays, sparse matrices, meshes)
- Store references in JSON that point to HDF5 datasets

Handles:
- Nested dicts and lists
- Pandas DataFrames (including object columns with lists, tuples)
- SimpleFaultSurface objects (via their mesh data)
- Sparse matrices (CSR, CSC, COO, DOK, LIL formats)
- Standard Python types (int, float, str, bool, None)
- Numpy arrays and scalars
"""

import json
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def serialize(data, filepath):
    """
    Serialize fault data structure to an HDF5 file.
    """
    with h5py.File(filepath, 'w') as f:
        blob_counter = [0]  # mutable counter for nested function

        def store_blob(arr):
            """Store a numpy array as an HDF5 dataset, return reference key."""
            key = f"blob_{blob_counter[0]}"
            blob_counter[0] += 1
            f.create_dataset(
                key, data=arr, compression='gzip', compression_opts=1
            )
            return key

        json_structure = _to_json_structure(data, store_blob)
        f.attrs['structure'] = json.dumps(json_structure)


def deserialize(filepath, raw_surfaces=False):
    """
    Deserialize fault data structure from an HDF5 file.
    """
    with h5py.File(filepath, 'r') as f:
        json_structure = json.loads(f.attrs['structure'])

        def load_blob(key):
            """Load a numpy array from an HDF5 dataset."""
            return f[key][:]

        return _from_json_structure(json_structure, load_blob, raw_surfaces)


def _to_json_structure(item, store_blob):
    """Convert an item to a JSON-serializable structure, storing blobs as needed."""

    if item is None:
        return None

    if isinstance(item, bool):
        return {'_t': 'bool', 'v': item}

    if isinstance(item, (int, np.integer)):
        return int(item)

    if isinstance(item, (float, np.floating)):
        v = float(item)
        if np.isnan(v):
            return {'_t': 'float', 'v': 'nan'}
        if np.isinf(v):
            return {'_t': 'float', 'v': 'inf' if v > 0 else '-inf'}
        return v

    if isinstance(item, str):
        return item

    if isinstance(item, tuple):
        return {
            '_t': 'tuple',
            'v': [_to_json_structure(x, store_blob) for x in item],
        }

    if isinstance(item, np.ndarray):
        if item.dtype == object:
            # Object arrays: store as list
            return {
                '_t': 'ndarray_obj',
                'shape': list(item.shape),
                'v': [
                    _to_json_structure(x, store_blob) for x in item.flatten()
                ],
            }
        else:
            # Numeric arrays: store as blob
            return {
                '_t': 'ndarray',
                'blob': store_blob(item),
                'dtype': str(item.dtype),
            }

    if sparse.issparse(item):
        return _sparse_to_json(item, store_blob)

    if isinstance(item, dict):
        # Check for non-string keys
        has_complex_keys = any(not isinstance(k, str) for k in item.keys())
        if has_complex_keys:
            return {
                '_t': 'dict_complex',
                'items': [
                    [
                        _to_json_structure(k, store_blob),
                        _to_json_structure(v, store_blob),
                    ]
                    for k, v in item.items()
                ],
            }
        return {k: _to_json_structure(v, store_blob) for k, v in item.items()}

    if isinstance(item, list):
        return [_to_json_structure(x, store_blob) for x in item]

    if isinstance(item, pd.DataFrame):
        return _dataframe_to_json(item, store_blob)

    if _is_simple_fault_surface(item):
        return _surface_to_json(item, store_blob)

    raise TypeError(f"Cannot serialize type: {type(item)}")


def _from_json_structure(item, load_blob, raw_surfaces=False):
    """Convert a JSON structure back to Python objects."""

    if item is None:
        return None

    if isinstance(item, bool):
        return item

    if isinstance(item, (int, float)):
        return item

    if isinstance(item, str):
        return item

    if isinstance(item, list):
        return [_from_json_structure(x, load_blob, raw_surfaces) for x in item]

    if isinstance(item, dict):
        if '_t' not in item:
            # Regular dict with string keys
            return {
                k: _from_json_structure(v, load_blob, raw_surfaces)
                for k, v in item.items()
            }

        t = item['_t']

        if t == 'bool':
            return item['v']

        if t == 'float':
            v = item['v']
            if v == 'nan':
                return float('nan')
            if v == 'inf':
                return float('inf')
            if v == '-inf':
                return float('-inf')

        if t == 'tuple':
            return tuple(
                _from_json_structure(x, load_blob, raw_surfaces)
                for x in item['v']
            )

        if t == 'ndarray':
            return load_blob(item['blob'])

        if t == 'ndarray_obj':
            shape = tuple(item['shape'])
            flat = [
                _from_json_structure(x, load_blob, raw_surfaces)
                for x in item['v']
            ]
            # Create empty object array and fill it to avoid numpy expanding nested lists
            arr = np.empty(len(flat), dtype=object)
            for i, val in enumerate(flat):
                arr[i] = val
            return arr.reshape(shape)

        if t == 'sparse':
            return _sparse_from_json(item, load_blob)

        if t == 'dict_complex':
            return {
                _from_json_structure(
                    k, load_blob, raw_surfaces
                ): _from_json_structure(v, load_blob, raw_surfaces)
                for k, v in item['items']
            }

        if t == 'dataframe':
            return _dataframe_from_json(item, load_blob, raw_surfaces)

        if t == 'SimpleFaultSurface':
            return _surface_from_json(item, load_blob, raw_surfaces)

    raise ValueError(f"Unknown structure: {item}")


def _sparse_to_json(mat, store_blob):
    """Convert sparse matrix/array to JSON structure."""
    # Determine format and whether it's array or matrix
    class_name = type(mat).__name__

    if 'csr' in class_name:
        fmt = 'csr'
    elif 'csc' in class_name:
        fmt = 'csc'
    elif 'coo' in class_name:
        fmt = 'coo'
    elif 'dok' in class_name:
        fmt = 'dok'
    elif 'lil' in class_name:
        fmt = 'lil'
    elif 'dia' in class_name:
        fmt = 'dia'
    elif 'bsr' in class_name:
        fmt = 'bsr'
    else:
        fmt = 'coo'

    # Track if it's an array (new style) vs matrix (old style)
    is_array = 'array' in class_name

    coo = mat.tocoo()
    return {
        '_t': 'sparse',
        'fmt': fmt,
        'is_array': is_array,
        'shape': list(coo.shape),
        'dtype': str(coo.dtype),
        'data': store_blob(coo.data),
        'row': store_blob(coo.row),
        'col': store_blob(coo.col),
    }


def _sparse_from_json(item, load_blob):
    """Reconstruct sparse matrix/array from JSON structure."""
    shape = tuple(item['shape'])
    dtype = np.dtype(item['dtype'])
    fmt = item['fmt']
    is_array = item.get('is_array', False)  # default False for backward compat

    data = load_blob(item['data'])
    row = load_blob(item['row'])
    col = load_blob(item['col'])

    # Create COO first, then convert
    if is_array:
        coo = sparse.coo_array((data, (row, col)), shape=shape, dtype=dtype)
        converters = {
            'csr': coo.tocsr,
            'csc': coo.tocsc,
            'coo': lambda: coo,
            'dok': coo.todok,
            'lil': coo.tolil,
            'dia': coo.todia,
            'bsr': coo.tobsr,
        }
    else:
        coo = sparse.coo_matrix((data, (row, col)), shape=shape, dtype=dtype)
        converters = {
            'csr': coo.tocsr,
            'csc': coo.tocsc,
            'coo': lambda: coo,
            'dok': coo.todok,
            'lil': coo.tolil,
            'dia': coo.todia,
            'bsr': coo.tobsr,
        }

    return converters.get(fmt, lambda: coo)()


def _dataframe_to_json(df, store_blob):
    """Convert DataFrame to JSON structure."""
    columns_data = {}

    for col in df.columns:
        series = df[col]

        # Check if it's a simple numeric column
        if series.dtype in (
            np.float64,
            np.float32,
            np.int64,
            np.int32,
            np.bool_,
        ):
            columns_data[col] = {
                '_t': 'ndarray',
                'blob': store_blob(series.values),
                'dtype': str(series.dtype),
            }
        else:
            # Object column or other - store as list
            columns_data[col] = [
                _to_json_structure(x, store_blob) for x in series.tolist()
            ]

    return {
        '_t': 'dataframe',
        'columns': list(df.columns),
        'index': _to_json_structure(df.index.tolist(), store_blob),
        'index_name': df.index.name,
        'data': columns_data,
    }


def _dataframe_from_json(item, load_blob, raw_surfaces=False):
    """Reconstruct DataFrame from JSON structure."""
    columns = item['columns']
    index = _from_json_structure(item['index'], load_blob, raw_surfaces)
    index_name = item['index_name']

    data = {}
    for col in columns:
        col_data = item['data'][col]
        if isinstance(col_data, dict) and col_data.get('_t') == 'ndarray':
            data[col] = load_blob(col_data['blob'])
        else:
            data[col] = [
                _from_json_structure(x, load_blob, raw_surfaces)
                for x in col_data
            ]

    df = pd.DataFrame(data, index=index, columns=columns)
    df.index.name = index_name
    return df


def _is_simple_fault_surface(item):
    """Check if item is a SimpleFaultSurface."""
    return type(item).__name__ == 'SimpleFaultSurface'


def _surface_to_json(surface, store_blob):
    """Convert SimpleFaultSurface to JSON structure."""
    mesh = surface.mesh
    result = {
        '_t': 'SimpleFaultSurface',
        'lons': store_blob(mesh.lons),
        'lats': store_blob(mesh.lats),
    }
    if mesh.depths is not None:
        result['depths'] = store_blob(mesh.depths)
    return result


def _surface_from_json(item, load_blob, raw_surfaces=False):
    """Reconstruct SimpleFaultSurface from JSON structure."""
    lons = load_blob(item['lons'])
    lats = load_blob(item['lats'])
    depths = load_blob(item['depths']) if 'depths' in item else None

    if raw_surfaces:
        return {
            '_type': 'SimpleFaultSurface',
            'lons': lons,
            'lats': lats,
            'depths': depths,
        }

    from openquake.hazardlib.geo.surface.simple_fault import SimpleFaultSurface
    from openquake.hazardlib.geo.mesh import RectangularMesh

    mesh = RectangularMesh(lons, lats, depths=depths)
    return SimpleFaultSurface(mesh)


# Convenience aliases
def save(data, filepath):
    """Alias for serialize()."""
    serialize(data, filepath)


def load(filepath, raw_surfaces=False):
    """Alias for deserialize()."""
    return deserialize(filepath, raw_surfaces=raw_surfaces)
