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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openquake.hazardlib.sourcewriter import write_source_model

from openquake.fnm.exporter import make_multifault_source
from openquake.fnm.inversion.fermi_importer import read_rup_csv

test_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

fgj_name = os.path.join(test_data_dir, "motagua_faults.geojson")


settings = {
    "general": {"fname_sections": fgj_name, "subsection_size": [-0.5, -1.0]},
    "ruptures": {"aspect_ratios": [0, 100], "magnitude_scaling_rel": "generic"},
    "connections": {
        "min_distance_between_subsections": {"threshold_distance": 10.0},
        "only_connections_on_edge": {},
    },
    "sections": {"subsections_size": [10, 10]},
}

from openquake.fnm.inversion.fermi_importer import (
    build_info_from_faults,
    build_system_of_equations,
    rup_df_to_rupture_dicts,
    subsection_df_to_fault_dicts,
)

from openquake.fnm.rupture_filtering import (
    get_rupture_plausibilities,
    filter_proportionally_to_plausibility,
)

from openquake.fnm.inversion.simulated_annealing import simulated_annealing

from openquake.fnm.inversion.plots import (
    plot_soln_slip_rates,
    plot_soln_mfd,
)

motagua_rup_df_file = os.path.join(test_data_dir, "motagua_test_ruptures.csv")
motagua_subsec_df_file = os.path.join(
    test_data_dir, "motagua_test_subsections.csv"
)
motagua_rup_plaus_df_file = os.path.join(
    test_data_dir, "motagua_test_rupture_plausibilities.csv"
)
keep_df_file = os.path.join(test_data_dir, "motagua_test_ruptures_kept.csv")
lhs_file = os.path.join(test_data_dir, "motagua_test_lhs.csv")
rhs_file = os.path.join(test_data_dir, "motagua_test_rhs.csv")
errs_file = os.path.join(test_data_dir, "motagua_test_errs.csv")
x_file = os.path.join(test_data_dir, "motagua_test_slip_rates.csv")


def test_load_faults_and_do_inversion():
    rup_fault_data, subsec_df, rup_df = build_info_from_faults(
        fgj_name, settings=settings
    )

    rup_df_ = read_rup_csv(motagua_rup_df_file)
    subsec_df_ = pd.read_csv(motagua_subsec_df_file, index_col=0)

    # pd.testing.assert_frame_equal(rup_df, rup_df_,
    #                              check_exact=False, atol=1e-1,
    #                              rtol=1e-1)
    # pd.testing.assert_frame_equal(subsec_df, subsec_df_)

    rup_plaus = get_rupture_plausibilities(rup_df)

    rup_plaus_ = read_rup_csv(motagua_rup_plaus_df_file)
    # pd.testing.assert_frame_equal(rup_plaus, rup_plaus_)

    keep_df = filter_proportionally_to_plausibility(
        rup_df,
        plausibility=rup_plaus.total,
        seed=5318008,
    )

    keep_df_ = read_rup_csv(keep_df_file)
    # pd.testing.assert_frame_equal(keep_df, keep_df_)

    faults = subsection_df_to_fault_dicts(subsec_df)
    ruptures = rup_df_to_rupture_dicts(keep_df)

    # with open(os.path.join(test_data_dir, "motagua_faults.json"), "w") as f:
    #    json.dump(faults, f)

    # with open(os.path.join(test_data_dir, "motagua_ruptures.json"), "w") as f:
    #    json.dump(ruptures, f)

    lhs, rhs, err = build_system_of_equations(keep_df, subsec_df)

    lhs_ = np.loadtxt(lhs_file)
    rhs_ = np.loadtxt(rhs_file)
    err_ = np.loadtxt(errs_file)

    # np.testing.assert_allclose(lhs.todense(), lhs_)
    # np.testing.assert_allclose(rhs, rhs_)
    # np.testing.assert_allclose(err, err_)

    x_, misfit_hist = simulated_annealing(
        lhs, rhs, weights=err, max_iters=int(1e6), accept_norm=1e-20, seed=69420
    )

    x__ = np.loadtxt(x_file)
    # np.testing.assert_allclose(x_, x__)

    rups_out = keep_df
    rups_out["occurrence_rate"] = x_

    mfs = make_multifault_source(
        rup_fault_data["fault_system"],
        rups_out,
        investigation_time=1.0,
        infer_occur_rates=True,
    )

    if not os.path.exists(os.path.join(test_data_dir, "ssm")):
        os.mkdir(os.path.join(test_data_dir, "ssm"))

    write_source_model(
        os.path.join(test_data_dir, "ssm", "mfs.xml"),
        [mfs],
        investigation_time=1.0,
    )

    # export_subsections(
    #    rup_fault_data["fault_system"],
    #    fname=os.path.join(test_data_dir, "motagua_subsections_out.geojson"),
    # )
    # export_ruptures(
    #    rup_fault_data["ruptures_single_section_indexes"],
    #    rup_fault_data["ruptures_single_section"],
    #    rup_fault_data["fault_system"],
    #    rup_fault_data["magnitudes"],
    #    rates=x_,
    #    fname=os.path.join(test_data_dir, "motagua_ruptures_out.geojson"),
    # )

    plt.figure()
    plot_soln_slip_rates(
        x_,
        subsec_df.slip_rate,
        lhs,
        errs=subsec_df.slip_rate_err,
        units="mm/yr",
    )
    plt.title("Observed and modeled slip rates")

    plt.figure()
    plot_soln_mfd(x_, ruptures)
    plt.title("Solution MFD")

    plt.show()
