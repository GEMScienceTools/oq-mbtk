# ------------------- The OpenQuake Model Building Toolkit --------------------
# ------------------- FERMI: Fault nEtwoRks ModellIng -------------------------
# Copyright (C) 2023 GEM Foundation
#         .-.
#        /    \                                        .-.
#        | .`. ;    .--.    ___ .-.     ___ .-. .-.   ( __)
#        | |(___)  /    \  (   )   \   (   )   '   \  (''")
#        | |_     |  .-. ;  | ' .-. ;   |  .-.  .-. ;  | |
#       (   __)   |  | | |  |  / (___)  | |  | |  | |  | |
#        | |      |  |/  |  | |         | |  | |  | |  | |
#        | |      |  ' _.'  | |         | |  | |  | |  | |
#        | |      |  .'.-.  | |         | |  | |  | |  | |
#        | |      '  `-' /  | |         | |  | |  | |  | |
#       (___)      `.__.'  (___)       (___)(___)(___)(___)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import os
import json
import pathlib
import unittest

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString

import openquake.hazardlib as hz

from openquake.fnm.inversion.utils import (
    geom_from_fault_trace,
    project_faults_and_polies,
    lines_in_polygon,
    get_rupture_displacement,
    weighted_mean,
    b_mle,
    get_a_b,
    slip_vector_azimuth,
    check_fault_in_poly,
    faults_in_polies,
    get_rup_poly_fracs,
    rup_df_to_rupture_dicts,
    subsection_df_to_fault_dicts,
    get_rupture_regions,
    _nearest,
    make_fault_mfd,
    get_mag_counts,
    get_mfd_occurrence_rates,
    get_mfd_moment,
    get_mfd_uncertainties,
    make_cumulative,
    set_single_fault_rupture_rates_by_mfd,
    set_single_fault_rup_rates,
    _get_surface_moment_rate,
    get_fault_moment_rate,
    _get_fault_by_id,
    get_ruptures_on_fault_df,
    get_ruptures_on_fault,
    make_rup_fault_lookup,
    get_rup_rates_from_fault_slip_rates,
    get_earthquake_fault_distances,
    get_on_fault_likelihood,
    get_soln_slip_rates,
    point_to_triangle_distance,
    calculate_tri_mesh_distances,
    rescale_mfd,
)

from openquake.fnm.all_together_now import build_fault_network

from openquake.fnm.tests.inversion.simple_test_data import (
    rup_A,
    rup_B,
    rup_C,
    rup_D,
    f1,
    f2,
    simple_test_rups,
    simple_test_faults,
    simple_test_fault_adjacence,
)

HERE = pathlib.Path(__file__).parent.absolute()
test_data_dir = HERE / ".." / 'data'

lil_fault_file = os.path.join(test_data_dir, "lil_test_faults.geojson")

guatemala_fault_file = os.path.join(test_data_dir, "guatemala_faults.geojson")
guatemala_fault_network = build_fault_network(
    fault_geojson=guatemala_fault_file,
    settings={'calculate_rates_from_slip_rates': False},
)


def test_geom_from_fault_trace():
    with open(lil_fault_file) as f:
        gj = json.load(f)

    geom = geom_from_fault_trace(gj['features'][0]['geometry']['coordinates'])
    ggeom = LineString(
        [
            Point(*xy)
            for xy in zip(
                [-122.6737, -122.6966, -122.76783, -122.81553, -122.86513],
                [45.48704, 45.52225, 45.58416, 45.62554, 45.66822],
            )
        ]
    )

    assert geom == geom


def test_project_faults_and_polies():
    pass


def test_lines_in_polygon():
    pass


def test_get_rupture_displacement():
    D = get_rupture_displacement(6.0, 100.0)
    np.testing.assert_approx_equal(D, 0.35063, significant=3)


def test_weighted_mean():
    w_mean = weighted_mean([80.0, 90.0], [20, 30])
    assert w_mean == 86.0


def test_b_mle():
    mags = np.array([6.0, 6.0, 6.2, 6.8, 7.0])
    b = b_mle(mags)
    np.testing.assert_approx_equal(b, 0.18095603412635491, significant=3)


def test_get_a_b():
    mags = np.array([6.0, 6.0, 6.2, 6.8, 7.0])
    a, b = get_a_b(mags)
    np.testing.assert_approx_equal(a, -0.1792658504865239, significant=3)
    np.testing.assert_approx_equal(b, 0.18095603412635491, significant=3)


def test_slip_vector_azimuth():
    np.testing.assert_approx_equal(
        slip_vector_azimuth(90.0, 45, 90.0), 0.0, significant=3
    )

    np.testing.assert_approx_equal(
        slip_vector_azimuth(45.0, 45, 90.0), 315.0, significant=3
    )

    np.testing.assert_approx_equal(
        slip_vector_azimuth(45.0, 45, -90.0), 135.0, significant=3
    )

    np.testing.assert_approx_equal(
        slip_vector_azimuth(45.0, 45, 0.0), 225.0, significant=3
    )


def check_fault_in_poly():
    pass


def test_get_mfd_moment_from_mfd_object():
    min_mag = 0.0
    max_mag = 7.0
    corner_mag = 6.5
    bin_width = 0.01
    b_val = 1.0
    moment = 1.0e16
    mfd = hz.mfd.TaperedGRMFD.from_moment(
        min_mag, max_mag, corner_mag, bin_width, b_val, moment
    )

    calc_moment = get_mfd_moment(mfd)

    np.testing.assert_approx_equal(moment, calc_moment, significant=3)


def test_get_mag_counts():
    rups = [rup_A, rup_B, rup_C, rup_D]
    mag_counts_default = get_mag_counts(rups)
    assert mag_counts_default == {6.0: 2, 6.5: 1, 7.0: 1}

    mag_counts_incremental = get_mag_counts(rups, cumulative=False)
    assert mag_counts_incremental == {6.0: 2, 6.5: 1, 7.0: 1}

    mag_counts_cumulative = get_mag_counts(rups, cumulative=True)
    assert mag_counts_cumulative == {6.0: 4, 6.5: 2, 7.0: 1}


class Test3Faults(unittest.TestCase):
    def setUp(self):
        test_data_dir = HERE / ".." / 'data'
        fgj_name = os.path.join(test_data_dir, "motagua_3_faults.geojson")


def test_check_faults_in_polies():
    poly_file = os.path.join(test_data_dir, "guatemala_regions.geojson")
    poly_gdf = gpd.read_file(poly_file)

    fault_poly_membership = faults_in_polies(
        guatemala_fault_network['faults'], poly_gdf, fault_id_key="fid"
    )
    assert fault_poly_membership == {
        'ccaf015': ['1'],
        'ccaf016': ['1'],
        'ccaf017': ['1'],
        'ccaf018': ['1'],
        'ccaf019': ['1'],
        'ccaf020': ['1'],
        'ccaf021': ['3'],
        'ccaf027': ['2', '3'],
        'ccaf022': ['2'],
        'ccaf023': ['2'],
        'ccaf121': ['2'],
        'ccaf127': ['3'],
        'ccaf128': ['3'],
        'ccaf129': ['3'],
        'ccaf130': ['3'],
        'ccaf131': ['3'],
        'ccaf135': ['1'],
        'ccaf134': ['1'],
        'ccaf148': ['1'],
        'ccaf149': ['1'],
    }


def test_get_rupture_regions():
    poly_file = os.path.join(test_data_dir, "guatemala_regions.geojson")
    poly_gdf = gpd.read_file(poly_file)
    rup_regions = get_rupture_regions(
        guatemala_fault_network['rupture_df_keep'],
        guatemala_fault_network['subfault_df'],
        poly_gdf,
        fault_key='subfaults',
    )

    # return rup_regions


def test_nearest():
    vals = [0.0, 1.0, 2.0, 3.0]

    assert _nearest(-1.0, vals) == 0.0
    assert _nearest(1.1, vals) == 1.0
    assert _nearest(10, vals) == 3.0


def test_make_fault_mfd():
    fault = guatemala_fault_network['faults'][0]
    trunc_mfd = make_fault_mfd(
        fault,
        mfd_type='TruncatedGRMFD',
        b_val=1.0,
        seismic_fraction=1.0,
        min_mag=5.0,
        max_mag=8.0,
        bin_width=0.1,
        moment_rate=None,
    )

    trunc_mfd_ = hz.mfd.TruncatedGRMFD.from_moment(
        5.0, 8.0, 0.1, 1.0, get_fault_moment_rate(fault)
    )

    assert trunc_mfd.__dict__ == trunc_mfd_.__dict__


def test_get_mag_counts():
    pass


def test_get_mfd_occurrence_rates():
    pass


def test_get_mfd_moment():
    pass


def test_get_mfd_uncertainties():
    pass


def test_make_cumulative():
    pass


def test_set_single_fault_rupture_rates_by_mfd():
    pass


def test_set_single_fault_rup_rates():
    pass


def test_get_surface_moment_rate():
    pass


def test_get_fault_moment_rate():
    pass


def test__get_fault_by_id():
    pass


def test_get_ruptures_on_fault_df():
    pass


def test_get_ruptures_on_fault():
    pass


def test_make_rup_fault_lookup():
    pass


def test_get_rup_rates_from_fault_slip_rates():
    pass


def test_get_earthquake_fault_distances():
    pass


def test_get_on_fault_likelihood():
    pass


def test_get_soln_slip_rates():
    pass


def test_point_to_triangle_distance():
    pass


def test_calculate_tri_mesh_distances():
    pass


def test_rescale_mfd():
    pass
