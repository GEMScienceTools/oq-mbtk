# coding: utf-8

import copy
import numpy as np
import numpy.typing as npt

from pyproj import Proj
from scipy.spatial.distance import cdist

from openquake.hazardlib.geo.mesh import Mesh
from openquake.hazardlib.geo.utils import plane_fit
from openquake.hazardlib.geo.geodetic import geodetic_distance, azimuth

from openquake.fnm.section import get_subsection


def get_connections(fsys: list, binm: np.ndarray, criteria: dict) -> list:
    """
    Computes the connections between a list of surfaces each one representing
    a section.

    :param fsys:
            A fault system. See :module:`openquake.fnm.fault_system`
    :param binm:
        A binary matrix indicating the combinations of sections in the fault
        system that should be considered in this analysis
    :param criteria:
        A dictionary containing the criteria for filtering our ruptures.
    :returns:
        An instance of :class:`numpy.ndarray` where each row contains the
        following information:
            - index first section
            - index second section
            - index row (upper left corner) in cells units first subsection
            - index column (upper left corner) in cells units
            - size rupture in number of cells along strike
            - size rupture in number of cells along dip
            - index row (upper left corner) in cells units second subsection
            - index column (upper left corner) in cells units
            - size rupture in number of cells along strike
            - size rupture in number of cells along dip
    """
    all_conns = []
    all_dists = []
    all_angls = []

    # Loop through the sections
    for i_sec_a in np.arange(0, len(fsys)):
        # Loop through the sections
        for i_sec_b in np.arange(i_sec_a + 1, len(fsys)):
            # Check the binary matrix. If 0, the bounding boxes of the two
            # subsections are not within the threshold distance so no detailed
            # analyses are necessary
            if binm[i_sec_a, i_sec_b] == 0:
                continue

            # Get subsections. The first variable contains the indexes of
            # the upper left corner and the second one the size (in cells) of
            # a subsection (along strike and along dip). The meshes have shape
            # 13 x 25 which means 12 x 24 cells
            mesh_a = fsys[i_sec_a][0].mesh
            sbs_a = fsys[i_sec_a][1]

            mesh_b = fsys[i_sec_b][0].mesh
            sbs_b = fsys[i_sec_b][1]

            # Get connections between subsections in sections
            conns, dists, angles = get_connections_between_subs(
                sbs_a, sbs_b, mesh_a, mesh_b, criteria)

            # Checking
            if len(conns):
                conns[:, 0] = i_sec_a
                conns[:, 1] = i_sec_b
                all_conns.extend(list(conns))
                all_dists.extend(list(dists))
                all_angls.extend(list(angles))

    # From lists to numpy arrays
    all_conns = np.array(all_conns)
    all_dists = np.array(all_dists)
    all_angls = np.array(all_angls)

    # A posteriori filtering connections
    criteria_distance = criteria.get('min_distance_between_subsections', {})
    if len(all_conns) and criteria_distance.get('shortest_only', True):
        idxs = filter_connections(all_conns, all_dists)
        all_conns = all_conns[idxs]
        all_dists = all_dists[idxs]
        all_angls = all_angls[idxs]

    # if len(all_conns) == 0:
    #    breakpoint()

    return all_conns, all_dists, all_angls


def filter_connections(conns: npt.ArrayLike, dists: npt.ArrayLike):
    """
    :param conns:
        An array with a raw list of connections found
    :param dists:
        An array with the
    :returns:

    """

    # Find the unique combinations of sections IDs i.e. all the connections
    # with the same section IDs
    unique_secs_idxs = np.vstack(list({tuple(e[0:2]) for e in conns}))

    # Find the combination of subsections with the shortest distance
    oidxs = []
    for unq in unique_secs_idxs:
        idx_1 = np.where(
            np.logical_and(conns[:, 0] == unq[0], conns[:, 1] == unq[1])
        )[0]
        idx_2 = np.argmin(dists[idx_1])
        oidxs.append(idx_1[idx_2])

    # Return a array with the unique of combination of sections
    return oidxs


def get_connections_between_subs(sbs_a, sbs_b, mesh_a, mesh_b, criteria):
    """
    Returns connections between the subsections representing two sections

    :param sbs_a:
        The first output of `split_into_subsections`
    :param nc_a:
        The second output of `split_into_subsections`
    :mesh_a:
        The mesh representing the surface of the first section
    :mesh_b:
        The mesh representing the surface of the second section
    :returns:
        A tuple of two :class:`numpy.ndarray` instances. The first array
        contains the connections and the second one contains the distances.
    """
    conns = []

    # Flatten the arrays describing the subsections
    sbs_a_nr = sbs_a.shape[0]
    sbs_a_nc = sbs_a.shape[1]
    sbs_b_nr = sbs_b.shape[0]
    sbs_b_nc = sbs_b.shape[1]

    # Reshaping the subsection matrices
    sbs_a = np.reshape(sbs_a, (sbs_a_nr * sbs_a_nc, -1))
    sbs_b = np.reshape(sbs_b, (sbs_b_nr * sbs_b_nc, -1))

    # Define the distance matrix [km]
    dstmtx = np.ones((len(sbs_a), len(sbs_b))) * 1e5

    # Iterate through the subsections in the first section.
    for i_ss_a, ss_a in enumerate(sbs_a):
        # Get the mesh for the current subsection
        ss_mesh_a = get_subsection(mesh_a, ss_a)

        # Create a line representing the top edge of the rupture. If the mesh
        # representing this subsection does not contain a sufficient number of
        # nodes the line is set to None.
        tidx = np.isfinite(ss_mesh_a.array[0, 0, :])
        if np.sum(tidx) < 2:
            continue

        # Iterate through the subsections in the second section.
        for i_ss_b, ss_b in enumerate(sbs_b):
            # Get mesh
            ss_mesh_b = get_subsection(mesh_b, ss_b)

            # Here we should apply the various criteria. For the time being
            # we do something very simple
            dst = ss_mesh_a.get_min_distance(ss_mesh_b)[0]

            # Fill the distance matrix
            dstmtx[i_ss_a, i_ss_b] = dst

            # Compute the dihedral angle and the angle betweent the lines
            # passing through the top of the sections.
            angle_dih, angle_top = get_angles(ss_mesh_a, ss_mesh_b)

            # Initialize the dictionary with the results of the requested
            # checks
            checks = {}
            for k in criteria.keys():
                checks[k] = False

            # Check minimum distance between subsections
            key = "min_distance_between_subsections"
            if key in criteria:
                if dst < criteria[key]["threshold_distance"]:
                    checks[key] = True
                else:
                    continue

            # Check if subsections are on the edges
            key = "only_connections_on_edge"
            if key in criteria:
                tidx_a = np.unravel_index(i_ss_a, (sbs_a_nr, sbs_a_nc))
                tidx_b = np.unravel_index(i_ss_b, (sbs_b_nr, sbs_b_nc))

                cond_a = (
                    (tidx_a[0] == 0)
                    or (tidx_a[0] == sbs_a_nr - 1)
                    or (tidx_a[1] == 0)
                    or (tidx_a[1] == sbs_a_nc - 1)
                )
                cond_b = (
                    (tidx_b[0] == 0)
                    or (tidx_b[0] == sbs_b_nr - 1)
                    or (tidx_b[1] == 0)
                    or (tidx_b[1] == sbs_b_nc - 1)
                )
                if cond_a and cond_b:
                    checks[key] = True

            # Checking connection angle
            key = "min_connection_angle"
            if key in criteria:
                if angle_top > criteria[key]["threshold_angle"]:
                    checks[key] = True
                else:
                    continue

            # If all the tests are passing
            if np.all(np.array([checks[k] for key in checks.keys()])):
                conns.append(
                    [
                        0,
                        0,
                        ss_a[0],
                        ss_a[1],
                        ss_a[2],
                        ss_a[3],
                        ss_b[0],
                        ss_b[1],
                        ss_b[2],
                        ss_b[3],
                    ]
                )

    # Exclude connections that are not between the two subsections at the
    # shortest distance
    if len(conns) > 0:
        out, out_dst, out_ang = filter_jumps_conns(conns, mesh_a, mesh_b)
        if len(out.shape) < 2:
            out = np.expand_dims(out, axis=0)
            out_dst = np.expand_dims(out_dst, axis=0)
            out_ang = np.expand_dims(out_ang, axis=0)
    else:
        out = conns
        out_dst = dst
        out_ang = angle_top

    return out, out_dst, out_ang


def filter_jumps_conns(conns, mesh_a, mesh_b):
    """
    Filters the connections between two sections. For the time being it
    selects the connection with the shortest distance between the two sections.

    :param conns:
        An iterable containing the connections between sections `a` and `b`
    :param mesh_a:
        The mesh representing the first section
    :param mesh_b:
        The mesh representing the second section
    :returns:
        A triple containing three arrays with the selected subset of
        connections, their shortest distances and, angles.
    """

    # Find the 'side' of the meshes with the closest distance
    i_a, i_b, dst_min = get_idxs_closest_points(mesh_a, mesh_b)

    alla = check_point_on_edge(np.array(i_a), mesh_a)
    allb = check_point_on_edge(np.array(i_b), mesh_b)
    alla = [int(d) for d in str(bin(alla))[2:]]
    allb = [int(d) for d in str(bin(allb))[2:]]
    alla = _pad(4, alla)
    allb = _pad(4, allb)

    # Work on all the connections between subsections
    mdists = []
    for conn in conns:
        s_msh_a = mesh_a[
            conn[2]:conn[2] + conn[5], conn[3]:conn[3] + conn[4]
        ]
        s_msh_b = mesh_b[
            conn[6]:conn[6] + conn[9], conn[7]:conn[7] + conn[8]
        ]
        idx_a, idx_b, dst = get_idxs_closest_points(s_msh_a, s_msh_b)
        mdists.append(dst)

        # Binary representation of the edges of each subsection close to the
        # other subsection
        chka = check_point_on_edge(np.array(idx_a), s_msh_a)
        chka = [int(d) for d in str(bin(chka))[2:]]
        chkb = check_point_on_edge(np.array(idx_b), s_msh_b)
        chkb = [int(d) for d in str(bin(chkb))[2:]]
        chka = _pad(4, chka)
        chkb = _pad(4, chkb)

        # Find neighbors
        """
        nei_a = check_neighbors(mesh_a, conn[2:6])
        nei_a = [int(d) for d in str(bin(nei_a))[2:]]
        nei_b = check_neighbors(mesh_b, conn[6:10])
        nei_b = [int(d) for d in str(bin(nei_b))[2:]]
        nei_a = _pad(4, nei_a)
        nei_b = _pad(4, nei_b)
        """

    # Find the index of the connection with the shortest distance
    idx = np.argmin(mdists)

    # Get the angle between the two planes
    conn = conns[idx]
    s_msh_a = mesh_a[conn[2]:conn[2] + conn[5], conn[3]:conn[3] + conn[4]]
    s_msh_b = mesh_b[conn[6]:conn[6] + conn[9], conn[7]:conn[7] + conn[8]]
    angle_dih, angle_top = get_angles(s_msh_a, s_msh_b)

    # Outputs
    oconn = np.array(conns[idx])
    odsts = np.array(mdists[idx])
    oagls = np.array(angle_top)
    return oconn, odsts, oagls


def get_angles(s_msh_a, s_msh_b):
    """
    Computes the dihedral angle between two planes. See for example
    https://en.wikipedia.org/wiki/Dihedral_angle and the angle between the
    traces of the traces of the two subsections involved.

    :param s_msh_a:
        A mesh
    :param s_msh_b:
        A mesh
    :returns:
        Two floats defining the dihedral angle and the angle between the two
        lines passing through the top of the two subsections.
    """

    # Get the closest points on the two meshes
    idx_a, idx_b, dst = get_idxs_closest_points(s_msh_a, s_msh_b)

    # Find the plane equation for the two subsections
    in_a = np.reshape(s_msh_a.array, (3, -1)).T
    in_b = np.reshape(s_msh_b.array, (3, -1)).T

    m_lon = np.mean(in_a[:, 0])
    m_lat = np.mean(in_a[:, 1])
    proj = Proj(proj="lcc", lon_0=m_lon, lat_1=m_lat - 10.0,
                lat_2=m_lat + 10.0)

    in_ap = copy.copy(in_a)
    in_bp = copy.copy(in_b)
    in_ap[:, 0], in_ap[:, 1] = proj(in_a[:, 0], in_a[:, 1])
    in_bp[:, 0], in_bp[:, 1] = proj(in_b[:, 0], in_b[:, 1])
    in_ap[:, 0:2] = in_ap[:, 0:2] / 1000  # to km as the depth
    in_bp[:, 0:2] = in_bp[:, 0:2] / 1000  # to km as the depth

    # Fit a plane on the surface of the subsection
    pnt_a, cos_a = plane_fit(in_ap)
    pnt_b, cos_b = plane_fit(in_bp)
    num = np.sum([a * b for a, b in zip(cos_a, cos_b)])
    den1 = np.sqrt(np.sum([a**2 for a in cos_a]))
    den2 = np.sqrt(np.sum([a**2 for a in cos_b]))
    dih_ang = np.abs(num) / (den1 * den2)

    # Compute the angle between traces
    azi_trace_a = azimuth(
        s_msh_a.lons[0, 0],
        s_msh_a.lats[0, 0],
        s_msh_a.lons[0, -1],
        s_msh_a.lats[0, -1],
    )
    azi_trace_b = azimuth(
        s_msh_b.lons[0, 0],
        s_msh_b.lats[0, 0],
        s_msh_b.lons[0, -1],
        s_msh_b.lats[0, -1],
    )

    # The two sections point in the same halfspace
    if np.abs((azi_trace_a - azi_trace_b) % 360) < 90:
        pass

    # Find the intersection point. See https://tinyurl.com/3b9n388t
    x1, y1 = proj(s_msh_a.lons[0, 0], s_msh_a.lats[0, 0])
    x2, y2 = proj(s_msh_a.lons[0, -1], s_msh_a.lats[0, -1])
    x3, y3 = proj(s_msh_b.lons[0, 0], s_msh_b.lats[0, 0])
    x4, y4 = proj(s_msh_b.lons[0, -1], s_msh_b.lats[0, -1])
    num = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Compute the angle between the lines passing through the top of the two
    # subsections
    trace_ang = 0.0
    if np.abs(den) > 1e-3:
        px = num / den
        num = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        py = num / den
        gpx, gpy = proj(px, py, inverse=True)

        # Find the azimuth between the intersection and the vertexes of each
        # trace
        azi_a = azimuth(
            gpx,
            gpy,
            [s_msh_a.lons[0, 0], s_msh_a.lons[0, -1]],
            [s_msh_a.lats[0, 0], s_msh_a.lats[0, -1]],
        )
        azi_b = azimuth(
            gpx,
            gpy,
            [s_msh_b.lons[0, 0], s_msh_b.lons[0, -1]],
            [s_msh_b.lats[0, 0], s_msh_b.lats[0, -1]],
        )

        # Compute the angle
        trace_ang = 180.0 - abs(abs(azi_a[0] - azi_b[0]) - 180)

        chk_a = 180.0 - abs(abs(azi_a[0] - azi_a[1]) - 180)
        chk_b = 180.0 - abs(abs(azi_b[0] - azi_b[1]) - 180)

        if chk_a > 90.0 or chk_b > 90.0:
            # Intersection is on one of the two traces
            trace_ang = 180.0 - abs(abs(azi_trace_a - azi_trace_b) - 180)

    return np.rad2deg(np.arccos(dih_ang)), trace_ang


def _pad(final_len, ilst):
    return ((final_len - len(ilst)) * [0] + ilst)[:final_len]


def check_neighbors(mesh: npt.ArrayLike, cell):
    """
    Check the subsection neighbors around `cell`.

    :param mesh:
        A :class:`numpy.ndarray` instance.
    :param cell:
        The indexes defining the size of the cell i.e. the subsection.
    :param out:
        A scalar
    """
    out = 0
    if cell[0] > 0:
        out += 1
    if cell[1] > 0:
        out += 8
    if cell[1] + cell[2] < mesh.shape[1] - 1:
        out += 2
    if cell[0] + cell[3] < mesh.shape[0] - 1:
        out += 4
    return out


def check_point_on_edge(idxs, mesh):
    """
    Check if the point specified by the index in `idxs` is on the edge of the
    mesh.
    We assign to each edge of the mesh a value:
        - Top = 1
        - Right = 2
        - Bottom = 4
        - Left = 8
    The returned value corresponds to the sum of the values assigned to each
    edge. Examples:
        - If the point is on the top edge the returned value is 1
        - If the point is at the bottom-left corner the returned value is 8+4

    :param idxs:
        An array with 2 columns and 'n' rows.
    :param mesh:
        A :class:`openquake.hazardlib.geo.mesh.Mesh` instance
    :returns:
        An integer that corresponds to the sum of the values for each of the
        four edges.
    """
    out = np.zeros_like((idxs.shape[0]))
    out[idxs[0] == 0] += 1
    out[idxs[1] == 0] += 2
    out[idxs[0] == mesh.shape[0] - 1] += 4
    out[idxs[1] == mesh.shape[1] - 1] += 8
    return out


def get_idxs_closest_points(mesha, meshb):
    """
    Compute for each of the two meshes the index of the closest point to the
    other mesh.

    :param mesha:
        A :class:`openquake.hazardlib.geo.mesh.Mesh` instance
    :param meshb:
        A :class:`openquake.hazardlib.geo.mesh.Mesh` instance
    :returns:
        Two tuples with the indexes of the closest points on the two meshes
        and the corresponding distance [km]
    """

    # Compute distances
    dists = cdist(mesha.xyz, meshb.xyz)

    # Find indexes
    i_closest_a = dists.min(axis=1).argmin()
    i_closest_b = dists.min(axis=0).argmin()
    idx_a = np.unravel_index(i_closest_a, mesha.shape)
    idx_b = np.unravel_index(i_closest_b, meshb.shape)

    return idx_a, idx_b, dists.min()


def get_jump_data(ssa: Mesh, ssb: Mesh):
    """
    Computes the characteristics of the jump along the shortest trajectory
    between two subsections (i.e. meshes). Note that we are assuming that the
    subsections traces are (almost) straight.

    :param ssa:
        A :class:`openquake.hazardlib.geo.mesh.Mesh` instance
    :param mesh:
        A :class:`openquake.hazardlib.geo.mesh.Mesh` instance
    """

    # TODO this is an implementation that does not work (since it does not
    # take into account the position of the subsection wrt to the section)

    tips_a = np.zeros((4, 2))
    tips_b = np.zeros((4, 2))

    # Tips first sub-section
    tips_a[0, 0] = tips_a[2, 0] = ssa.array[0, 0, 0]
    tips_a[1, 0] = tips_a[3, 0] = ssa.array[0, 0, -1]
    tips_a[0, 1] = tips_a[2, 1] = ssa.array[1, 0, 0]
    tips_a[1, 1] = tips_a[3, 1] = ssa.array[1, 0, -1]

    # Tips second sub-section
    tips_b[0, 0] = tips_b[2, 0] = ssb.array[0, 0, 0]
    tips_b[0, 1] = tips_b[3, 0] = ssb.array[0, 0, -1]
    tips_b[1, 1] = tips_b[2, 1] = ssb.array[1, 0, 0]
    tips_b[0, 1] = tips_b[3, 1] = ssb.array[1, 0, -1]

    # Calculate distances between the tips of the two subsections
    dsts = geodetic_distance(
        tips_a[:, 0], tips_a[:, 1], tips_b[:, 0], tips_b[:, 1]
    )

    # Index of the shortest distance
    idx = np.argmin(dsts)

    import matplotlib.pyplot as plt

    _ = plt.figure()
    plt.plot(ssa.array[0, 0, :], ssa.array[1, 0, :], "-xr")
    plt.plot(ssb.array[0, 0, :], ssb.array[1, 0, :], "-xb")
    plt.plot(tips_a[idx, 0], tips_a[idx, 1], "or", mfc="none")
    plt.plot(tips_b[idx, 0], tips_b[idx, 1], "ob", mfc="none")
    plt.show()
