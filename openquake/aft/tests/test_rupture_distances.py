import pytest
import numpy as np

from openquake.hazardlib.geo import Polygon, Point
from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.hazardlib.scalerel.wc1994 import WC1994
from openquake.hazardlib.source.area import AreaSource
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.geo.nodalplane import NodalPlane

#from .rupture_distances import (
from openquake.aft.rupture_distances import (
    RupDistType,
    calc_min_source_dist,
    get_close_source_pairs,
    calc_pairwise_distances,
    min_reduce,
    stack_sequences,
    split_rows,
    get_min_rup_dists,
    check_dists_by_mag,
    filter_dists_by_mag,
    get_rup_dist_pairs,
    process_source_pair,
    calc_rupture_adjacence_dict_all_sources,
    prep_source_data,
)

area_source_1 = AreaSource(
    source_id="s1",
    name="s1",
    tectonic_region_type="ActiveShallowCrust",
    mfd=TruncatedGRMFD(
        min_mag=4.6, max_mag=8.0, bin_width=0.2, a_val=1.0, b_val=1.0
    ),
    magnitude_scaling_relationship=WC1994(),
    rupture_aspect_ratio=1.0,
    temporal_occurrence_model=PoissonTOM,
    upper_seismogenic_depth=0.0,
    lower_seismogenic_depth=30.0,
    nodal_plane_distribution=PMF([(1.0, NodalPlane(0.0, 90, 180.0))]),
    hypocenter_distribution=PMF([(1.0, 15.0)]),
    polygon=Polygon(
        [
            Point(0.0, 0.0, 0.0),
            Point(1.0, 0.0, 0.0),
            Point(1.0, 1.0, 0.0),
            Point(0.0, 1.0, 0),
            Point(0.0, 0.0, 0.0),
        ]
    ),
    area_discretization=15.0,
    rupture_mesh_spacing=5.0,
)

area_source_2 = AreaSource(
    source_id="s2",
    name="s2",
    tectonic_region_type="ActiveShallowCrust",
    mfd=TruncatedGRMFD(
        min_mag=4.6, max_mag=8.0, bin_width=0.2, a_val=1.0, b_val=1.0
    ),
    magnitude_scaling_relationship=WC1994(),
    rupture_aspect_ratio=1.0,
    temporal_occurrence_model=PoissonTOM,
    upper_seismogenic_depth=0.0,
    lower_seismogenic_depth=30.0,
    nodal_plane_distribution=PMF([(1.0, NodalPlane(0.0, 90, 180.0))]),
    hypocenter_distribution=PMF([(1.0, 15.0)]),
    polygon=Polygon(
        [
            Point(2.0, 0.0, 0.0),
            Point(2.0, -1.0, 0.0),
            Point(3.0, -1.0, 0.0),
            Point(3.0, 0.0, 0),
            Point(2.0, 0.0, 0.0),
        ]
    ),
    area_discretization=15.0,
    rupture_mesh_spacing=5.0,
)

area_source_3 = AreaSource(
    source_id="s3",
    name="s3",
    tectonic_region_type="ActiveShallowCrust",
    mfd=TruncatedGRMFD(
        min_mag=4.6, max_mag=8.0, bin_width=0.2, a_val=1.0, b_val=1.0
    ),
    magnitude_scaling_relationship=WC1994(),
    rupture_aspect_ratio=1.0,
    temporal_occurrence_model=PoissonTOM,
    upper_seismogenic_depth=0.0,
    lower_seismogenic_depth=30.0,
    nodal_plane_distribution=PMF([(1.0, NodalPlane(0.0, 90, 180.0))]),
    hypocenter_distribution=PMF([(1.0, 15.0)]),
    polygon=Polygon(
        [
            Point(4.0, 0.0, 0.0),
            Point(4.0, 1.0, 0.0),
            Point(5.0, 1.0, 0.0),
            Point(5.0, 0.0, 0),
            Point(4.0, 0.0, 0.0),
        ]
    ),
    area_discretization=15.0,
    rupture_mesh_spacing=5.0,
)


def test_calc_min_source_dist():
    dist = calc_min_source_dist(area_source_1, area_source_2)
    assert np.round(dist) == 111.0


def test_get_close_source_pairs_filter():
    close_source_pairs = get_close_source_pairs(
        [area_source_1, area_source_2, area_source_3], dist_threshold=150.0
    )

    close_source_pairs_answer = {
        ("s1", "s1"): 0.0,
        ("s2", "s1"): 111.19351532028067,
        ("s2", "s2"): 0.0,
        ("s3", "s2"): 111.19351532028064,
        ("s3", "s3"): 0.0,
    }

    pytest.approx(close_source_pairs, close_source_pairs_answer)


def test_get_close_source_pairs_no_filter():
    close_source_pairs = get_close_source_pairs(
        [area_source_1, area_source_2, area_source_3], dist_threshold=None
    )

    close_source_pairs_answer = {
        ("s1", "s1"): 0.0,
        ("s2", "s1"): 111.19351532028067,
        ("s2", "s2"): 0.0,
        ("s3", "s1"): 333.495874564696,
        ("s3", "s2"): 111.19351532028064,
        ("s3", "s3"): 0.0,
    }

    pytest.approx(close_source_pairs, close_source_pairs_answer)


def test_calc_pairwise_distances():
    v1 = np.array([[1000.0, 2000.0, 0.0], [1500.0, 2500.0, 0.0]])
    v2 = np.array([[2000.0, 2500.0, 0.0], [1000.0, 2000.0, 10.0]])

    pair_dists = calc_pairwise_distances(v1, v2)

    pair_dists_answer = np.array(
        [[1118.03398875, 10.0], [500.0, 707.17748833]]
    )

    np.testing.assert_array_almost_equal(pair_dists, pair_dists_answer)


def test_min_reduce():
    min_red_test_arr = np.array(
        [
            [10.0, 10.0, 0.0, 10.0, 10.0, 1.0, 10.0, 2.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 3.0, 10.0, 10.0, 4.0, 10.0, 10.0, 10.0, 5.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [6.0, 10.0, 10.0, 10.0, 10.0, 7.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 10.0],
        ]
    )

    row_inds = np.array([0, 2, 4])
    col_inds = np.array([0, 4, 6])

    reduced_min_answer = np.array(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
    )

    reduced_min = min_reduce(min_red_test_arr, row_inds, col_inds)

    numpy_reduce_min = np.minimum.reduceat(
        np.minimum.reduceat(min_red_test_arr, row_inds), col_inds, axis=1
    )

    # assert matches my expectations
    np.testing.assert_array_almost_equal(reduced_min, reduced_min_answer)

    # assert matches numpy
    np.testing.assert_array_almost_equal(reduced_min, numpy_reduce_min)


def test_stack_sequences():
    sequences = (
        [[0, 0], [0, 0], [0, 0]],
        [[1, 1], [1, 1]],
        [[2, 2], [2, 2], [2, 2]],
    )

    index_stack_answer = np.array([0, 3, 5], dtype=np.int32)
    value_stack_answer = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )

    idx_stack, val_stack = stack_sequences(sequences)

    np.testing.assert_array_equal(index_stack_answer, idx_stack)
    np.testing.assert_array_equal(value_stack_answer, val_stack)


def test_split_rows():
    lens = [2, 3, 7, 2, 3]

    rid_test = np.cumsum(lens[:-1])
    rid_test = np.insert(rid_test, 0, 0)
    xyz_test = np.vstack([np.ones((ll, 3)) * i for i, ll in enumerate(lens)])

    data_splits = split_rows(rid_test, xyz_test, 3)

    data_split_answer = {
        0: {
            "array_stack": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
            "row_idxs": np.array([0, 2]),
        },
        2: {
            "array_stack": np.array(
                [
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                ]
            ),
            "row_idxs": np.array([0]),
        },
        3: {
            "array_stack": np.array(
                [
                    [3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0],
                ]
            ),
            "row_idxs": np.array([0, 2]),
        },
    }

    for k in data_splits.keys():
        pytest.approx(data_splits[k], data_split_answer[k])

    assert data_splits.keys() == data_split_answer.keys()


def test_get_min_rup_dists_no_offset():

    dist_test_arr = np.array(
        [
            [10.0, 10.0, 0.0, 10.0, 10.0, 1.0, 10.0, 2.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 3.0, 10.0, 10.0, 4.0, 10.0, 10.0, 10.0, 5.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [6.0, 10.0, 10.0, 10.0, 10.0, 7.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 10.0],
        ]
    )

    row_inds = np.array([0, 2, 4])
    col_inds = np.array([0, 4, 6])

    min_rup_dists = get_min_rup_dists(dist_test_arr, row_inds, col_inds)

    min_rup_dists_answer = np.array(
        [
            (0, 0, 0.0),
            (0, 1, 1.0),
            (0, 2, 2.0),
            (1, 0, 3.0),
            (1, 1, 4.0),
            (1, 2, 5.0),
            (2, 0, 6.0),
            (2, 1, 7.0),
            (2, 2, 8.0),
        ],
        dtype=[("r1", "<i4"), ("r2", "<i4"), ("d", "<f4")],
    )

    np.testing.assert_array_equal(min_rup_dists, min_rup_dists_answer)


def test_get_min_rup_dists_offset():

    dist_test_arr = np.array(
        [
            [10.0, 10.0, 0.0, 10.0, 10.0, 1.0, 10.0, 2.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 3.0, 10.0, 10.0, 4.0, 10.0, 10.0, 10.0, 5.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [6.0, 10.0, 10.0, 10.0, 10.0, 7.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 10.0],
        ]
    )

    row_inds = np.array([0, 2, 4])
    col_inds = np.array([0, 4, 6])

    min_rup_dists = get_min_rup_dists(
        dist_test_arr, row_inds, col_inds, row_offset=5
    )

    min_rup_dists_answer = np.array(
        [
            (5, 0, 0.0),
            (5, 1, 1.0),
            (5, 2, 2.0),
            (6, 0, 3.0),
            (6, 1, 4.0),
            (6, 2, 5.0),
            (7, 0, 6.0),
            (7, 1, 7.0),
            (7, 2, 8.0),
        ],
        dtype=[("r1", "<i4"), ("r2", "<i4"), ("d", "<f4")],
    )

    np.testing.assert_array_equal(min_rup_dists, min_rup_dists_answer)


def test_check_dists_by_mag_1():

    dists = np.array([0.0, 10.0, 50.0, 1000.0, 2000.0])

    mags = np.array([9.0, 7.0, 6.0, 3.0, 2.0])

    np.testing.assert_array_equal(
        check_dists_by_mag(dists, mags),
        np.array([True, True, True, False, False]),
    )


def test_check_dists_by_mag_2():
    dists = np.array(
        [
            (0, 0, 0.0),
            (0, 1, 10.0),
            (0, 2, 200.0),
            (1, 0, 30.0),
            (1, 1, 400.0),
            (1, 2, 500.0),
            (2, 0, 60.0),
            (2, 1, 700.0),
            (2, 2, 800.0),
        ],
        dtype=[("r1", "<i4"), ("r2", "<i4"), ("d", "<f4")],
    )

    dist_vals = dists["d"]

    mags = np.array([3.0, 3.0, 3.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0])

    good_answer = [True, False, False, True, False, False, True, False, False]

    np.testing.assert_array_equal(
        check_dists_by_mag(dist_vals, mags), good_answer
    )


def test_filter_dists_by_mag():
    dists = np.array(
        [
            (0, 0, 0.0),
            (0, 1, 10.0),
            (0, 2, 200.0),
            (1, 0, 30.0),
            (1, 1, 400.0),
            (1, 2, 500.0),
            (2, 0, 60.0),
            (2, 1, 700.0),
            (2, 2, 800.0),
        ],
        dtype=[("r1", "<i4"), ("r2", "<i4"), ("d", "<f4")],
    )

    mags = np.array([3.0, 6.0, 7.0])
    filtered_dists = filter_dists_by_mag(dists, mags)

    filtered_dists_answer = np.array(
        [(0, 0, 0.0), (1, 0, 30.0), (2, 0, 60.0)], dtype=RupDistType
    )
    np.testing.assert_array_equal(filtered_dists, filtered_dists_answer)


def test_get_rup_dist_pairs():
    rup_df, source_groups = prep_source_data(
        [area_source_1, area_source_2, area_source_3]
    )
    rup_dist_pairs = get_rup_dist_pairs(
        "s1", "s2", rup_df, source_groups, dist_constant=4.0
    )
    rdp0 = np.array((11, 16, 223.67159), dtype=RupDistType)
    assert rdp0 == rup_dist_pairs[0]


def test_process_source_pair():
    rup_df, source_groups = prep_source_data(
        [area_source_1, area_source_2, area_source_3]
    )
    source_pair = ("s1", "s2")
    rup_adj_dict = {}

    process_source_pair(
        source_pair,
        rup_adj_dict,
        rup_df,
        source_groups,
        dist_constant=4.0,
    )
    rdp0 = np.array((11, 16, 223.67159), dtype=RupDistType)

    assert rup_adj_dict["s1"]["s2"][0] == rdp0


def test_calc_rupture_distance_dict_all_sources():

    sources = [area_source_1, area_source_2, area_source_3]
    rup_df, source_groups = prep_source_data(sources)
    source_pairs = get_close_source_pairs(sources)

    print(source_pairs)

    rup_adj_dict = calc_rupture_adjacence_dict_all_sources(
        source_pairs=source_pairs,
        rup_df=rup_df,
        source_groups=source_groups,
        dist_constant=4.0,
    )

    rdp0 = np.array((11, 16, 223.67159), dtype=RupDistType)
    assert rup_adj_dict["s1"]["s2"][0] == rdp0
