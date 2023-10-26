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


import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None

try:
    import pygmt
except:
    pygmt = None


def _get_mtx(mesh):
    grd = np.zeros((mesh.lons.size, 3))
    grd[:, 0] = mesh.lons.flatten()
    grd[:, 1] = mesh.lats.flatten()
    grd[:, 2] = mesh.depths.flatten()
    return grd


def _get_polydata(mesh, scl):
    """
    :param mesh:
        A :class:`openquake.hazardlib.geo.mesh.Mesh` instance
    :param scl:
        The vertical scaling factor
    """
    grd = _get_mtx(mesh)
    grd[:, 2] *= scl
    polydata = pv.PolyData(grd)
    return polydata


def _get_ul_polydata(mesh, scl, lab):
    """
    :param mesh:
        A :class:`openquake.hazardlib.geo.mesh.Mesh` instance
    :param scl:
        The vertical scaling factor
    """
    grd = _get_mtx(mesh)
    grd[:, 2] *= scl
    dat = np.array([grd[0, 0], grd[0, 1], grd[0, 2]])
    polydata = pv.PolyData(dat)
    polydata["ids"] = [lab]
    return polydata


def _plot_cell(pl, mesh, cell, scl, color="black"):
    # Plots a cell (e.g. a subsection) cobvering part of the mesh. Cell
    # contains 4 indexes indicating the upper left cell and the number of
    # mesh-cells along strike and dip composing it

    lo = mesh.lons
    la = mesh.lats
    dp = mesh.depths

    tmp = [
        [
            lo[cell[0], cell[1]],
            la[cell[0], cell[1]],
            dp[cell[0], cell[1]] * scl,
        ],
        [
            lo[cell[0] + cell[3], cell[1]],
            la[cell[0] + cell[3], cell[1]],
            dp[cell[0] + cell[3], cell[1]] * scl,
        ],
        [
            lo[cell[0] + cell[3], cell[1] + cell[2]],
            la[cell[0] + cell[3], cell[1] + cell[2]],
            dp[cell[0] + cell[3], cell[1] + cell[2]] * scl,
        ],
        [
            lo[cell[0], cell[1] + cell[2]],
            la[cell[0], cell[1] + cell[2]],
            dp[cell[0], cell[1] + cell[2]] * scl,
        ],
        [
            lo[cell[0], cell[1]],
            la[cell[0], cell[1]],
            dp[cell[0], cell[1]] * scl,
        ],
    ]

    tmp = np.array(tmp)
    pl.add_lines(np.array(tmp), color=color, connected=True)


def plot(meshes, **kwargs):
    pl = pv.Plotter()
    scl = 1.0 / 100

    # Plot meshes
    for i, mesh in enumerate(meshes):
        pdata = _get_polydata(mesh, scl)
        color = np.random.rand(3)
        pl.add_mesh(pdata.points, color=color, point_size=5, style="points")
        ldata = _get_ul_polydata(mesh, scl, f"{i}")
        pl.add_point_labels(ldata, "ids", point_size=1, font_size=20)

    if "rupture" in kwargs:
        # Retrieve the rupture, i.e. a :class:`numpy.ndarray` instance with
        # rows corresponding to the number of subsections
        rup = kwargs["rup"]
        for ssec in rup:
            pass

    if "fsys" in kwargs:

        for tmp in kwargs["fsys"]:
            surf = tmp[0]
            subs = tmp[1]
            msh = surf.mesh

            for i_row in range(subs.shape[0]):
                for i_col in range(subs.shape[1]):
                    _plot_cell(pl, msh, subs[i_row, i_col, :], scl)

    if "connections" in kwargs:
        conns = kwargs["connections"]
        conns = conns.astype(int)

        for conn in conns:
            msh = meshes[conn[0]]
            _plot_cell(pl, msh, conn[2:6], scl, color="red")

            msh = meshes[conn[1]]
            _plot_cell(pl, msh, conn[6:10], scl, color="red")

    # Final settings
    _ = pl.view_isometric()
    _ = pl.add_axes(line_width=5, labels_off=False)
    pl.set_viewup((0, 0, -1))
    pl.set_scale(zscale=-1)
    # pl.show_grid()

    # Marker
    # marker = pv.create_axes_marker()
    # pl.add_actor(marker)
    pl.show(interactive=True)


def plot_profiles(profiles, trace):
    min_lo = 400
    min_la = 400
    max_lo = -400
    max_la = -400

    # Computing extent of the region
    dlt = 0.5
    for pro in profiles:
        min_lo = np.min([min_lo, np.min(pro.coo[:, 0])]) - dlt
        min_la = np.min([min_la, np.min(pro.coo[:, 1])]) - dlt
        max_lo = np.max([max_lo, np.max(pro.coo[:, 0])]) + dlt
        max_la = np.max([max_la, np.max(pro.coo[:, 1])]) + dlt
    region = [min_lo, max_lo, min_la, max_la]
    print(region)

    fig = pygmt.Figure()
    fig.basemap(region=region, projection="M15c", frame=True)
    fig.coast(land="lightgray", water="skyblue")
    fig.plot(x=trace.coo[:, 0], y=trace.coo[:, 1], pen="0.2p,red")
    for pro in profiles:
        fig.plot(x=pro.coo[:, 0], y=pro.coo[:, 1], pen="0.2p,black")
    fig.show()
    breakpoint()
