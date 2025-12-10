"""
Comprehensive tests for fault_serializer module.

Tests cover:
- Basic Python types (int, float, str, bool, None, nan, inf)
- Collections (list, tuple, dict with various key types)
- NumPy arrays (numeric and object dtypes)
- Pandas DataFrames (numeric and object columns)
- SciPy sparse matrices (all formats)
- OpenQuake SimpleFaultSurface objects
- Full fault network roundtrip

Run with: pytest test_fault_network_serializer.py -v
Run fast tests only: pytest test_fault_network_serializer.py -v -m "not slow"
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy import sparse

import openquake.fnm.fault_network_serializer as fs


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_h5_file():
    """Provide a temporary HDF5 file path that is cleaned up after the test."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        filepath = f.name
    yield filepath
    if os.path.exists(filepath):
        os.unlink(filepath)


@pytest.fixture
def fault_network():
    """
    Build a fault network from test data.

    Requires openquake.fnm to be installed and test data to be present.
    """
    pytest.importorskip("openquake.fnm", reason="openquake.fnm not installed")
    from openquake.fnm.all_together_now import build_fault_network

    settings = {
        "subsection_size": [12.0, 10.0],
        "lower_seis_depth": 10.0,
        "filter_by_plausibility": False,
    }
    test_data_dir = os.path.join(os.path.dirname(__file__), "data")
    fault_file = os.path.join(test_data_dir, "lil_test_faults.geojson")

    return build_fault_network(settings=settings, fault_geojson=fault_file)


# =============================================================================
# Helper functions
# =============================================================================


def assert_surfaces_equal(surf1, surf2):
    """Assert two SimpleFaultSurface objects are equal by comparing their meshes."""
    assert_array_equal(surf1.mesh.lons, surf2.mesh.lons)
    assert_array_equal(surf1.mesh.lats, surf2.mesh.lats)
    if surf1.mesh.depths is not None or surf2.mesh.depths is not None:
        assert_array_equal(surf1.mesh.depths, surf2.mesh.depths)


def assert_fault_dicts_equal(fault1, fault2):
    """Assert two fault dictionaries are equal, handling surfaces specially."""
    assert (
        fault1.keys() == fault2.keys()
    ), f"Keys differ: {fault1.keys()} vs {fault2.keys()}"

    for k, v in fault1.items():
        if k == 'surface':
            assert_surfaces_equal(v, fault2[k])
        elif isinstance(v, np.ndarray):
            assert_array_equal(v, fault2[k])
        elif isinstance(v, float) and np.isnan(v):
            assert np.isnan(fault2[k])
        else:
            assert (
                v == fault2[k]
            ), f"Mismatch for key '{k}': {v} != {fault2[k]}"


def assert_dataframes_equal(df1, df2):
    """Assert two DataFrames are equal, handling object columns with surfaces."""
    assert list(df1.columns) == list(df2.columns), "Column names differ"
    assert len(df1) == len(df2), "Row counts differ"
    assert df1.index.tolist() == df2.index.tolist(), "Indices differ"

    for col in df1.columns:
        for i in range(len(df1)):
            v1 = df1[col].iloc[i]
            v2 = df2[col].iloc[i]

            if hasattr(v1, 'mesh'):  # SimpleFaultSurface
                assert_surfaces_equal(v1, v2)
            elif isinstance(v1, np.ndarray):
                assert_array_equal(v1, v2)
            elif isinstance(v1, float) and np.isnan(v1):
                assert np.isnan(v2)
            elif isinstance(v1, (list, tuple)):
                assert v1 == v2, f"Mismatch in column '{col}' row {i}"
            else:
                assert (
                    v1 == v2
                ), f"Mismatch in column '{col}' row {i}: {v1} != {v2}"


def roundtrip(data, temp_file, raw_surfaces=False):
    """Serialize and deserialize data, returning the loaded result."""
    fs.serialize(data, temp_file)
    return fs.deserialize(temp_file, raw_surfaces=raw_surfaces)


# =============================================================================
# Basic type tests
# =============================================================================


class TestBasicTypes:
    """Tests for basic Python types."""

    def test_none(self, temp_h5_file):
        data = {'value': None}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] is None

    def test_bool_true(self, temp_h5_file):
        data = {'value': True}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] is True
        assert isinstance(loaded['value'], bool)

    def test_bool_false(self, temp_h5_file):
        data = {'value': False}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] is False
        assert isinstance(loaded['value'], bool)

    def test_int(self, temp_h5_file):
        data = {'value': 42}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == 42
        assert isinstance(loaded['value'], int)

    def test_int_negative(self, temp_h5_file):
        data = {'value': -123456}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == -123456

    def test_int_large(self, temp_h5_file):
        data = {'value': 10**18}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == 10**18

    def test_float(self, temp_h5_file):
        data = {'value': 3.14159}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == pytest.approx(3.14159)

    def test_float_negative(self, temp_h5_file):
        data = {'value': -273.15}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == pytest.approx(-273.15)

    def test_float_nan(self, temp_h5_file):
        data = {'value': float('nan')}
        loaded = roundtrip(data, temp_h5_file)
        assert np.isnan(loaded['value'])

    def test_float_inf(self, temp_h5_file):
        data = {'value': float('inf')}
        loaded = roundtrip(data, temp_h5_file)
        assert np.isinf(loaded['value'])
        assert loaded['value'] > 0

    def test_float_neg_inf(self, temp_h5_file):
        data = {'value': float('-inf')}
        loaded = roundtrip(data, temp_h5_file)
        assert np.isinf(loaded['value'])
        assert loaded['value'] < 0

    def test_string(self, temp_h5_file):
        data = {'value': 'hello world'}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == 'hello world'

    def test_string_empty(self, temp_h5_file):
        data = {'value': ''}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == ''

    def test_string_unicode(self, temp_h5_file):
        data = {'value': '日本語 émojis 🌍'}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == '日本語 émojis 🌍'

    def test_string_special_chars(self, temp_h5_file):
        data = {'value': 'line1\nline2\ttab"quote\'apostrophe'}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == 'line1\nline2\ttab"quote\'apostrophe'


class TestNumpyScalars:
    """Tests for numpy scalar types."""

    def test_numpy_int32(self, temp_h5_file):
        data = {'value': np.int32(42)}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == 42
        assert isinstance(loaded['value'], int)

    def test_numpy_int64(self, temp_h5_file):
        data = {'value': np.int64(-999)}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == -999

    def test_numpy_float32(self, temp_h5_file):
        data = {'value': np.float32(3.14)}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == pytest.approx(3.14, rel=1e-5)

    def test_numpy_float64(self, temp_h5_file):
        data = {'value': np.float64(2.718281828)}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == pytest.approx(2.718281828)


# =============================================================================
# Collection tests
# =============================================================================


class TestLists:
    """Tests for list serialization."""

    def test_empty_list(self, temp_h5_file):
        data = {'value': []}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == []

    def test_list_of_ints(self, temp_h5_file):
        data = {'value': [1, 2, 3, 4, 5]}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == [1, 2, 3, 4, 5]

    def test_list_of_floats(self, temp_h5_file):
        data = {'value': [1.1, 2.2, 3.3]}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == pytest.approx([1.1, 2.2, 3.3])

    def test_list_of_strings(self, temp_h5_file):
        data = {'value': ['a', 'b', 'c']}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == ['a', 'b', 'c']

    def test_list_mixed_types(self, temp_h5_file):
        data = {'value': [1, 'two', 3.0, None, True]}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'][0] == 1
        assert loaded['value'][1] == 'two'
        assert loaded['value'][2] == 3.0
        assert loaded['value'][3] is None
        assert loaded['value'][4] is True

    def test_nested_lists(self, temp_h5_file):
        data = {'value': [[1, 2], [3, 4], [5, 6]]}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == [[1, 2], [3, 4], [5, 6]]

    def test_deeply_nested_lists(self, temp_h5_file):
        data = {'value': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]


class TestTuples:
    """Tests for tuple serialization."""

    def test_empty_tuple(self, temp_h5_file):
        data = {'value': ()}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == ()
        assert isinstance(loaded['value'], tuple)

    def test_tuple_of_ints(self, temp_h5_file):
        data = {'value': (1, 2, 3)}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == (1, 2, 3)
        assert isinstance(loaded['value'], tuple)

    def test_tuple_mixed_types(self, temp_h5_file):
        data = {'value': (1, 'two', 3.0)}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == (1, 'two', 3.0)
        assert isinstance(loaded['value'], tuple)

    def test_nested_tuples(self, temp_h5_file):
        data = {'value': ((0, 0), (0, 1), (1, 0))}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == ((0, 0), (0, 1), (1, 0))

    def test_tuple_vs_list_preserved(self, temp_h5_file):
        """Ensure tuples and lists are distinguished after roundtrip."""
        data = {'tuple_val': (1, 2), 'list_val': [1, 2]}
        loaded = roundtrip(data, temp_h5_file)
        assert isinstance(loaded['tuple_val'], tuple)
        assert isinstance(loaded['list_val'], list)


class TestDicts:
    """Tests for dictionary serialization."""

    def test_empty_dict(self, temp_h5_file):
        data = {'value': {}}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == {}

    def test_dict_string_keys(self, temp_h5_file):
        data = {'value': {'a': 1, 'b': 2, 'c': 3}}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == {'a': 1, 'b': 2, 'c': 3}

    def test_dict_nested(self, temp_h5_file):
        data = {'value': {'outer': {'inner': {'deep': 42}}}}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value']['outer']['inner']['deep'] == 42

    def test_dict_int_keys(self, temp_h5_file):
        """Dicts with int keys should roundtrip correctly."""
        data = {'value': {1: 'one', 2: 'two', 3: 'three'}}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == {1: 'one', 2: 'two', 3: 'three'}

    def test_dict_tuple_keys(self, temp_h5_file):
        """Dicts with tuple keys should roundtrip correctly."""
        data = {'value': {(0, 0): 'origin', (1, 0): 'right', (0, 1): 'up'}}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == {
            (0, 0): 'origin',
            (1, 0): 'right',
            (0, 1): 'up',
        }

    def test_dict_mixed_values(self, temp_h5_file):
        data = {
            'value': {
                'int': 42,
                'float': 3.14,
                'string': 'hello',
                'list': [1, 2, 3],
                'nested': {'a': 1},
            }
        }
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value']['int'] == 42
        assert loaded['value']['float'] == pytest.approx(3.14)
        assert loaded['value']['string'] == 'hello'
        assert loaded['value']['list'] == [1, 2, 3]
        assert loaded['value']['nested'] == {'a': 1}


# =============================================================================
# NumPy array tests
# =============================================================================


class TestNumpyArrays:
    """Tests for NumPy array serialization."""

    def test_1d_array(self, temp_h5_file):
        data = {'value': np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        loaded = roundtrip(data, temp_h5_file)
        assert_array_equal(loaded['value'], data['value'])

    def test_2d_array(self, temp_h5_file):
        data = {'value': np.array([[1, 2, 3], [4, 5, 6]])}
        loaded = roundtrip(data, temp_h5_file)
        assert_array_equal(loaded['value'], data['value'])

    def test_3d_array(self, temp_h5_file):
        data = {'value': np.random.rand(3, 4, 5)}
        loaded = roundtrip(data, temp_h5_file)
        assert_array_almost_equal(loaded['value'], data['value'])

    def test_array_float32(self, temp_h5_file):
        data = {'value': np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        loaded = roundtrip(data, temp_h5_file)
        assert_array_almost_equal(loaded['value'], data['value'])

    def test_array_float64(self, temp_h5_file):
        data = {'value': np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        loaded = roundtrip(data, temp_h5_file)
        assert_array_equal(loaded['value'], data['value'])

    def test_array_int32(self, temp_h5_file):
        data = {'value': np.array([1, 2, 3], dtype=np.int32)}
        loaded = roundtrip(data, temp_h5_file)
        assert_array_equal(loaded['value'], data['value'])

    def test_array_int64(self, temp_h5_file):
        data = {'value': np.array([1, 2, 3], dtype=np.int64)}
        loaded = roundtrip(data, temp_h5_file)
        assert_array_equal(loaded['value'], data['value'])

    def test_array_bool(self, temp_h5_file):
        data = {'value': np.array([True, False, True, False])}
        loaded = roundtrip(data, temp_h5_file)
        assert_array_equal(loaded['value'], data['value'])

    def test_array_empty(self, temp_h5_file):
        data = {'value': np.array([])}
        loaded = roundtrip(data, temp_h5_file)
        assert_array_equal(loaded['value'], data['value'])

    def test_array_object_dtype(self, temp_h5_file):
        """Object arrays with mixed types."""
        data = {'value': np.array([[1, 2], [3, 4, 5], 'string'], dtype=object)}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'][0] == [1, 2]
        assert loaded['value'][1] == [3, 4, 5]
        assert loaded['value'][2] == 'string'

    def test_array_object_2d(self, temp_h5_file):
        """2D object array."""
        arr = np.empty((2, 3), dtype=object)
        arr[0, 0] = [1, 2]
        arr[0, 1] = [3, 4]
        arr[0, 2] = [5, 6]
        arr[1, 0] = (7, 8)
        arr[1, 1] = (9, 10)
        arr[1, 2] = (11, 12)
        data = {'value': arr}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'].shape == (2, 3)
        assert loaded['value'][0, 0] == [1, 2]
        assert loaded['value'][1, 2] == (11, 12)


# =============================================================================
# Pandas DataFrame tests
# =============================================================================


class TestDataFrames:
    """Tests for Pandas DataFrame serialization."""

    def test_simple_dataframe(self, temp_h5_file):
        data = {'df': pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})}
        loaded = roundtrip(data, temp_h5_file)
        pd.testing.assert_frame_equal(loaded['df'], data['df'])

    def test_dataframe_float_columns(self, temp_h5_file):
        data = {
            'df': pd.DataFrame(
                {
                    'x': [1.1, 2.2, 3.3],
                    'y': [4.4, 5.5, 6.6],
                }
            )
        }
        loaded = roundtrip(data, temp_h5_file)
        pd.testing.assert_frame_equal(loaded['df'], data['df'])

    def test_dataframe_string_column(self, temp_h5_file):
        data = {
            'df': pd.DataFrame(
                {
                    'name': ['Alice', 'Bob', 'Charlie'],
                    'age': [25, 30, 35],
                }
            )
        }
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['df']['name'].tolist() == ['Alice', 'Bob', 'Charlie']
        assert loaded['df']['age'].tolist() == [25, 30, 35]

    def test_dataframe_list_column(self, temp_h5_file):
        """DataFrame with list values in a column."""
        data = {
            'df': pd.DataFrame(
                {
                    'values': [[1, 2], [3, 4, 5], [6]],
                    'label': ['a', 'b', 'c'],
                }
            )
        }
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['df']['values'].tolist() == [[1, 2], [3, 4, 5], [6]]
        assert loaded['df']['label'].tolist() == ['a', 'b', 'c']

    def test_dataframe_tuple_column(self, temp_h5_file):
        """DataFrame with tuple values in a column."""
        data = {
            'df': pd.DataFrame(
                {
                    'position': [(0, 0), (0, 1), (1, 0)],
                    'value': [1, 2, 3],
                }
            )
        }
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['df']['position'].tolist() == [(0, 0), (0, 1), (1, 0)]

    def test_dataframe_custom_index(self, temp_h5_file):
        data = {'df': pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['df'].index.tolist() == ['x', 'y', 'z']

    def test_dataframe_named_index(self, temp_h5_file):
        df = pd.DataFrame({'a': [1, 2, 3]})
        df.index.name = 'my_index'
        data = {'df': df}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['df'].index.name == 'my_index'

    def test_dataframe_empty(self, temp_h5_file):
        data = {'df': pd.DataFrame()}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['df'].empty

    def test_dataframe_many_columns(self, temp_h5_file):
        """DataFrame with many columns of different types."""
        data = {
            'df': pd.DataFrame(
                {
                    'int_col': [1, 2, 3],
                    'float_col': [1.1, 2.2, 3.3],
                    'str_col': ['a', 'b', 'c'],
                    'list_col': [[1], [2], [3]],
                    'bool_col': [True, False, True],
                }
            )
        }
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['df']['int_col'].tolist() == [1, 2, 3]
        assert loaded['df']['float_col'].tolist() == pytest.approx(
            [1.1, 2.2, 3.3]
        )
        assert loaded['df']['str_col'].tolist() == ['a', 'b', 'c']
        assert loaded['df']['list_col'].tolist() == [[1], [2], [3]]
        assert loaded['df']['bool_col'].tolist() == [True, False, True]


# =============================================================================
# Sparse matrix tests
# =============================================================================


class TestSparseMatrices:
    """Tests for SciPy sparse matrix serialization."""

    def test_csr_matrix(self, temp_h5_file):
        mat = sparse.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
        data = {'mat': mat}
        loaded = roundtrip(data, temp_h5_file)
        assert sparse.isspmatrix_csr(loaded['mat'])
        assert_array_equal(loaded['mat'].toarray(), mat.toarray())

    def test_csc_matrix(self, temp_h5_file):
        mat = sparse.csc_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
        data = {'mat': mat}
        loaded = roundtrip(data, temp_h5_file)
        assert sparse.isspmatrix_csc(loaded['mat'])
        assert_array_equal(loaded['mat'].toarray(), mat.toarray())

    def test_coo_matrix(self, temp_h5_file):
        mat = sparse.coo_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
        data = {'mat': mat}
        loaded = roundtrip(data, temp_h5_file)
        assert sparse.isspmatrix_coo(loaded['mat'])
        assert_array_equal(loaded['mat'].toarray(), mat.toarray())

    def test_dok_matrix(self, temp_h5_file):
        mat = sparse.dok_matrix((3, 3), dtype=np.float64)
        mat[0, 1] = 1.5
        mat[2, 2] = 2.5
        data = {'mat': mat}
        loaded = roundtrip(data, temp_h5_file)
        assert sparse.isspmatrix_dok(loaded['mat'])
        assert_array_equal(loaded['mat'].toarray(), mat.toarray())

    def test_lil_matrix(self, temp_h5_file):
        mat = sparse.lil_matrix((3, 3), dtype=np.float64)
        mat[0, 1] = 1.5
        mat[2, 2] = 2.5
        data = {'mat': mat}
        loaded = roundtrip(data, temp_h5_file)
        assert sparse.isspmatrix_lil(loaded['mat'])
        assert_array_equal(loaded['mat'].toarray(), mat.toarray())

    def test_sparse_int_dtype(self, temp_h5_file):
        mat = sparse.csr_matrix([[1, 0, 2], [0, 0, 3]], dtype=np.int32)
        data = {'mat': mat}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['mat'].dtype == np.int32
        assert_array_equal(loaded['mat'].toarray(), mat.toarray())

    def test_sparse_empty(self, temp_h5_file):
        mat = sparse.csr_matrix((100, 100), dtype=np.float64)
        data = {'mat': mat}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['mat'].nnz == 0
        assert loaded['mat'].shape == (100, 100)

    def test_sparse_large(self, temp_h5_file):
        """Large sparse matrix."""
        mat = sparse.random(1000, 1000, density=0.01, format='csr')
        data = {'mat': mat}
        loaded = roundtrip(data, temp_h5_file)
        assert_array_almost_equal(loaded['mat'].toarray(), mat.toarray())


# =============================================================================
# SimpleFaultSurface tests
# =============================================================================

# Check if openquake is available
try:
    from openquake.hazardlib.geo.surface.simple_fault import SimpleFaultSurface
    from openquake.hazardlib.geo.mesh import RectangularMesh

    HAS_OPENQUAKE = True
except ImportError:
    HAS_OPENQUAKE = False


@pytest.mark.skipif(not HAS_OPENQUAKE, reason="OpenQuake not installed")
class TestSimpleFaultSurface:
    """Tests for OpenQuake SimpleFaultSurface serialization."""

    @pytest.fixture
    def simple_surface(self):
        """Create a simple test surface."""
        from openquake.hazardlib.geo.surface.simple_fault import (
            SimpleFaultSurface,
        )
        from openquake.hazardlib.geo.mesh import RectangularMesh

        lons = np.array([[-122.0, -122.1], [-122.0, -122.1], [-122.0, -122.1]])
        lats = np.array([[45.0, 45.0], [45.1, 45.1], [45.2, 45.2]])
        depths = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])

        mesh = RectangularMesh(lons, lats, depths=depths)
        return SimpleFaultSurface(mesh)

    def test_surface_roundtrip(self, temp_h5_file, simple_surface):
        data = {'surface': simple_surface}
        loaded = roundtrip(data, temp_h5_file)
        assert_surfaces_equal(simple_surface, loaded['surface'])

    def test_surface_raw_mode(self, temp_h5_file, simple_surface):
        """Test raw_surfaces=True returns dict instead of object."""
        data = {'surface': simple_surface}
        fs.serialize(data, temp_h5_file)
        loaded = fs.deserialize(temp_h5_file, raw_surfaces=True)

        assert isinstance(loaded['surface'], dict)
        assert loaded['surface']['_type'] == 'SimpleFaultSurface'
        assert_array_equal(loaded['surface']['lons'], simple_surface.mesh.lons)
        assert_array_equal(loaded['surface']['lats'], simple_surface.mesh.lats)
        assert_array_equal(
            loaded['surface']['depths'], simple_surface.mesh.depths
        )

    def test_surface_in_list(self, temp_h5_file, simple_surface):
        """Surface inside a list."""
        data = {'surfaces': [simple_surface, simple_surface]}
        loaded = roundtrip(data, temp_h5_file)
        assert len(loaded['surfaces']) == 2
        assert_surfaces_equal(simple_surface, loaded['surfaces'][0])
        assert_surfaces_equal(simple_surface, loaded['surfaces'][1])

    def test_surface_in_dict(self, temp_h5_file, simple_surface):
        """Surface inside a dict with other values."""
        data = {
            'fault': {
                'fid': 'f1',
                'rake': 90.0,
                'surface': simple_surface,
            }
        }
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['fault']['fid'] == 'f1'
        assert loaded['fault']['rake'] == 90.0
        assert_surfaces_equal(simple_surface, loaded['fault']['surface'])


# =============================================================================
# Complex structure tests
# =============================================================================


class TestComplexStructures:
    """Tests for complex nested structures similar to fault network data."""

    def test_fault_like_structure(self, temp_h5_file):
        """Test structure similar to actual fault data."""
        data = {
            'faults': [
                {
                    'fid': 'f1',
                    'net_slip_rate': 1.0,
                    'net_slip_rate_err': 0.5,
                    'rake': 135.0,
                    'trace': [[-122.6737, 45.48704], [-122.6966, 45.52225]],
                },
                {
                    'fid': 'f2',
                    'net_slip_rate': 1.0,
                    'net_slip_rate_err': 0.1,
                    'rake': 90.0,
                    'trace': [[-122.51594, 45.47618], [-122.66243, 45.47729]],
                },
            ],
            'subfaults': [
                [
                    {
                        'fid': 'f1',
                        'fault_position': (0, 0),
                        'subsec_id': 0,
                    },
                    {
                        'fid': 'f1',
                        'fault_position': (0, 1),
                        'subsec_id': 1,
                    },
                ],
                [
                    {
                        'fid': 'f2',
                        'fault_position': (0, 0),
                        'subsec_id': 0,
                    }
                ],
            ],
            'rupture_df': pd.DataFrame(
                {
                    'subfaults': [[0], [0, 1], [1], [2], [0, 1, 2]],
                    'faults': [['f1'], ['f1'], ['f1'], ['f2'], ['f1', 'f2']],
                    'mag': [6.1, 6.4, 6.1, 6.0, 6.5],
                }
            ),
            'dist_mat': sparse.csr_matrix(
                np.array(
                    [
                        [0, 0, 0, 5.2],
                        [0, 0, 0, 3.1],
                        [0, 0, 0, 0],
                        [5.2, 3.1, 0, 0],
                    ]
                )
            ),
            'multifault_inds': [[1, 3]],
        }

        loaded = roundtrip(data, temp_h5_file)

        # Verify faults
        assert len(loaded['faults']) == 2
        assert loaded['faults'][0]['fid'] == 'f1'
        assert loaded['faults'][1]['rake'] == 90.0

        # Verify subfaults
        assert len(loaded['subfaults']) == 2
        assert len(loaded['subfaults'][0]) == 2
        assert loaded['subfaults'][0][0]['fault_position'] == (0, 0)

        # Verify DataFrame
        assert loaded['rupture_df']['subfaults'].tolist() == [
            [0],
            [0, 1],
            [1],
            [2],
            [0, 1, 2],
        ]

        # Verify sparse matrix
        assert sparse.isspmatrix_csr(loaded['dist_mat'])

        # Verify multifault_inds
        assert loaded['multifault_inds'] == [[1, 3]]


# =============================================================================
# Full fault network roundtrip tests
# =============================================================================


class TestFaultNetworkRoundtrip:
    """
    Full roundtrip tests with actual fault network data.

    These tests require openquake.fnm to be installed and test data present.
    """

    def test_full_roundtrip(self, temp_h5_file, fault_network):
        """Test complete roundtrip of fault network data."""
        fs.serialize(fault_network, temp_h5_file)
        loaded = fs.deserialize(temp_h5_file)

        # Check top-level keys
        assert set(loaded.keys()) == set(fault_network.keys())

    def test_faults_roundtrip(self, temp_h5_file, fault_network):
        """Test faults list roundtrip."""
        fs.serialize(fault_network, temp_h5_file)
        loaded = fs.deserialize(temp_h5_file)

        assert len(loaded['faults']) == len(fault_network['faults'])

        for i, fault in enumerate(fault_network['faults']):
            assert_fault_dicts_equal(fault, loaded['faults'][i])

    def test_subfaults_roundtrip(self, temp_h5_file, fault_network):
        """Test subfaults nested list roundtrip."""
        fs.serialize(fault_network, temp_h5_file)
        loaded = fs.deserialize(temp_h5_file)

        assert len(loaded['subfaults']) == len(fault_network['subfaults'])

        for i, fault_subfaults in enumerate(fault_network['subfaults']):
            assert len(loaded['subfaults'][i]) == len(fault_subfaults)
            for j, subfault in enumerate(fault_subfaults):
                assert_fault_dicts_equal(subfault, loaded['subfaults'][i][j])

    def test_dataframes_roundtrip(self, temp_h5_file, fault_network):
        """Test DataFrame roundtrip."""
        fs.serialize(fault_network, temp_h5_file)
        loaded = fs.deserialize(temp_h5_file)

        # Check rupture_df
        if 'rupture_df' in fault_network:
            assert_dataframes_equal(
                fault_network['rupture_df'], loaded['rupture_df']
            )

        # Check subfault_df
        if 'subfault_df' in fault_network:
            assert_dataframes_equal(
                fault_network['subfault_df'], loaded['subfault_df']
            )

        # Check single_rup_df
        if 'single_rup_df' in fault_network:
            assert_dataframes_equal(
                fault_network['single_rup_df'], loaded['single_rup_df']
            )

    def test_sparse_matrices_roundtrip(self, temp_h5_file, fault_network):
        """Test sparse matrix roundtrip with format preservation."""
        fs.serialize(fault_network, temp_h5_file)
        loaded = fs.deserialize(temp_h5_file)

        if 'dist_mat' in fault_network:
            orig = fault_network['dist_mat']
            load = loaded['dist_mat']
            assert type(orig) == type(
                load
            ), f"Format mismatch: {type(orig)} vs {type(load)}"
            assert_array_almost_equal(orig.toarray(), load.toarray())

        if 'bin_dist_mat' in fault_network:
            orig = fault_network['bin_dist_mat']
            load = loaded['bin_dist_mat']
            assert type(orig) == type(
                load
            ), f"Format mismatch: {type(orig)} vs {type(load)}"
            assert_array_equal(orig.toarray(), load.toarray())

    def test_raw_surfaces_mode(self, temp_h5_file, fault_network):
        """Test that raw_surfaces=True works for full fault network."""
        fs.serialize(fault_network, temp_h5_file)
        loaded = fs.deserialize(temp_h5_file, raw_surfaces=True)

        # Check that surfaces are returned as dicts
        if 'faults' in loaded and len(loaded['faults']) > 0:
            if 'surface' in loaded['faults'][0]:
                surface = loaded['faults'][0]['surface']
                assert isinstance(surface, dict)
                assert surface['_type'] == 'SimpleFaultSurface'
                assert 'lons' in surface
                assert 'lats' in surface


# =============================================================================
# Edge case and error handling tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_long_string(self, temp_h5_file):
        data = {'value': 'x' * 100000}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == 'x' * 100000

    def test_deeply_nested_structure(self, temp_h5_file):
        """Test deeply nested dict/list structure."""
        data = {
            'level0': {'level1': {'level2': {'level3': {'level4': [1, 2, 3]}}}}
        }
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['level0']['level1']['level2']['level3']['level4'] == [
            1,
            2,
            3,
        ]

    def test_many_keys(self, temp_h5_file):
        """Test dict with many keys."""
        data = {f'key_{i}': i for i in range(1000)}
        loaded = roundtrip(data, temp_h5_file)
        assert len(loaded) == 1000
        assert loaded['key_500'] == 500

    def test_large_list(self, temp_h5_file):
        """Test large list."""
        data = {'value': list(range(10000))}
        loaded = roundtrip(data, temp_h5_file)
        assert loaded['value'] == list(range(10000))

    def test_unsupported_type_raises(self, temp_h5_file):
        """Test that unsupported types raise TypeError."""

        class CustomClass:
            pass

        data = {'value': CustomClass()}
        with pytest.raises(TypeError, match="Cannot serialize type"):
            fs.serialize(data, temp_h5_file)

    def test_file_overwrite(self, temp_h5_file):
        """Test that serializing to existing file overwrites it."""
        data1 = {'value': 'first'}
        data2 = {'value': 'second', 'extra': 42}

        fs.serialize(data1, temp_h5_file)
        fs.serialize(data2, temp_h5_file)

        loaded = fs.deserialize(temp_h5_file)
        assert loaded['value'] == 'second'
        assert loaded['extra'] == 42


# =============================================================================
# API tests
# =============================================================================


class TestAPI:
    """Tests for the module's public API."""

    def test_save_load_aliases(self, temp_h5_file):
        """Test that save/load are aliases for serialize/deserialize."""
        data = {'value': 42}
        fs.save(data, temp_h5_file)
        loaded = fs.load(temp_h5_file)
        assert loaded['value'] == 42

    def test_load_raw_surfaces_parameter(self, temp_h5_file):
        """Test that load() accepts raw_surfaces parameter."""
        data = {'value': 42}
        fs.save(data, temp_h5_file)
        loaded = fs.load(temp_h5_file, raw_surfaces=True)
        assert loaded['value'] == 42


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
