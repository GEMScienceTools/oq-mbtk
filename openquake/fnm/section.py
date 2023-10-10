#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.typing as npt
from openquake.hazardlib.geo.mesh import RectangularMesh, Mesh


def get_subsection(mesh: Mesh, sections_ul: np.array) -> RectangularMesh:
    """
    Given the mesh representing the surface of a section and subsection info
    returns a :class:`openquake.hazardlib.geo.mesh.Mesh` instance

    :param mesh:
        The mesh representing the surface of the section
    :param sections_ul:
        The indexes of the upper left corner of the mesh representing the
        surface of the subsection
    :param np.array:
        A vector with the number of cells along the strike and along the dip
        used to represent the section
    :returns:
        A mesh describing the surface of the subsection
    """
    ir = int(sections_ul[0])
    ic = int(sections_ul[1])
    nc_strike = int(sections_ul[2])
    nc_dip = int(sections_ul[3])
    tmp_mesh = RectangularMesh(
        lons=mesh.lons[ir : ir + nc_dip + 1, ic : ic + nc_strike + 1],
        lats=mesh.lats[ir : ir + nc_dip + 1, ic : ic + nc_strike + 1],
        depths=mesh.depths[ir : ir + nc_dip + 1, ic : ic + nc_strike + 1],
    )
    return tmp_mesh


def split_into_subsections(mesh, nc_stk=-1, nc_dip=-1) -> npt.ArrayLike:
    """
    This splits a mesh (we assume this is a mesh representing a kite surface)
    into a number of subsections.

    :param mesh:
        An :class:`openquake.hazardlib.geo.mesh.Mesh` instance
    :param nc_stk:
        Number of cells along strike defining a section
    :param nc_dip:
        Number of cells along dip defining a section. When equal to -1, the
        subsections will extend for the entire width of the surface.
    :returns:
        An array where each row contains the upper-left corner of each
        subsection (i.e. the length of this list is the number of subsections
        created), the number of cells along the strike and along the dip.
    """

    cells_row = mesh.shape[0] - 1
    cells_col = mesh.shape[1] - 1

    # Set the default number of cells when this info is not provided. We assume
    # that sections are rupturing the whole seismogenic layer. When negative,
    # the abs of the value is interpreted as the number of subsections along
    # the dip
    if nc_dip < 0:
        nc_dip = int(np.floor(cells_row / np.abs(nc_dip)))

    # If the number of cells along the strike is negative, that indicates the
    # factor that multipled by the number of cells along the dip gives gives
    # the number of cells along the strike
    if nc_stk < 0:
        nc_stk = max(int(np.floor(nc_dip * np.abs(nc_stk))), 1)

    # This collects the upper left corner of each section defined on this fault
    # surface. We add nc_dip-1 to make sure we incorporate the last subsection
    # that might not have the standard number of cells along strike and dip
    # `irow` and `icol` are the indexes of the cells used to the define the
    # subsections
    up_row = (
        cells_row
        if cells_row % nc_dip < 1
        else (cells_row // nc_dip) * nc_dip + nc_dip
    )

    up_col = (
        cells_col
        if cells_col % nc_stk < 1
        else (cells_col // nc_stk) * nc_stk + nc_stk
    )

    dlt_r = 0
    if cells_row % nc_dip > 0:
        dlt_r = 1
    dlt_c = 0
    if cells_col % nc_stk > 0:
        dlt_c = 1

    irows = list(np.arange(nc_dip, up_row + dlt_r + 1, nc_dip))
    icols = list(np.arange(nc_stk, up_col + dlt_c + 1, nc_stk))
    sections_ul = np.zeros((len(irows), len(icols), 4))

    for i, irow in enumerate(irows):
        for j, icol in enumerate(icols):
            tmp_stk = nc_stk
            low_c = icol - tmp_stk
            if icol > cells_col:
                low_c = sections_ul[i, j - 1, 1] + sections_ul[i, j - 1, 2]
                tmp_stk = cells_col - (low_c)

            tmp_dip = nc_dip
            low_r = irow - tmp_dip
            if irow > cells_row:
                low_r = sections_ul[i - 1, j, 0] + sections_ul[i - 1, j, 3]
                tmp_dip = cells_row - (low_r)

            if tmp_dip == 0 or tmp_stk == 0:
                raise ValueError("Section with one null dimension")

            sections_ul[i, j, :] = [low_r, low_c, tmp_stk, tmp_dip]

    return sections_ul.astype(int)
